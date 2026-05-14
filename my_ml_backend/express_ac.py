"""从 Doris 拉取字典，用 Aho-Corasick（AC）自动机在群聊文本里做多模式子串匹配。

本模块在 NER 预标注流程中的角色
--------------------------------
`model.py` 在 `predict` 时除正则规则外，还会用本模块构建的两台 AC 自动机扫描整条消息：
1. **快递单号 / 面单号**：字典来自 Doris 的 `expressNumber` 字段（见 `fetch_express_dictionary`），命中结果经 ``iter_matches`` 输出。
2. **买家昵称**：字典来自 Doris 的 `buyersNickname` 字段；同样用 AC 子串扫描，但经 ``iter_buyer_nickname_matches`` 增加**空白边界**过滤（避免无空白长串中的嵌入短词误标），再经 ``_resolve_spans`` 去重叠。

与「暴力对每个词在文本里 find」相比，AC 在一次从左到右扫描中同时报告词典中所有子串出现位置，复杂度近似线性于文本长度加命中数；买家昵称在 AC 之后另有业务过滤。

Aho-Corasick（摘要）
--------------------
Trie + 失配链 + 输出链；由 **pyahocorasick** 实现，``add_word`` 后 ``make_automaton()``，再用 ``iter(text)`` 扫描。

与 Label Studio 的衔接
----------------------
``iter_matches`` / ``iter_buyer_nickname_matches`` 返回的 ``(start, end, word)`` 中 ``end`` 为 **Python 切片右开区间**（即匹配子串为 ``text[start:end]``），供 ``model.py`` 转成 `HyperTextLabels` 等区域标注。
"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from decimal import Decimal
from typing import Any, List, Set, Tuple

logger = logging.getLogger("express_ac")

# pyahocorasick 为运行时依赖；若部署环境未安装，本模块降级为「无 AC」，由上层仅用正则等规则。
try:
    import ahocorasick
except ImportError:  # pragma: no cover
    ahocorasick = None

from express_doris import _qualified_table_sql

# ---------------------------------------------------------------------------
# 进程内单例：两台自动机 + 最近一次加载时间（单调时钟秒）
#
# 说明：快递与买家昵称共用 _last_success_monotonic / _last_attempt_monotonic 是历史设计；
# 任一侧成功/失败刷新都会更新这两个全局变量，因此「刷新间隔」在两侧交替加载时可能互相影响。
# 若将来要精确到「各自独立 TTL」，可拆成四元组时间戳；当前保持行为不变。
# ---------------------------------------------------------------------------
_express_automaton: Any | None = None
_buyer_nickname_automaton: Any | None = None
_last_success_monotonic: float = 0.0
_last_attempt_monotonic: float = 0.0
_load_lock = threading.Lock()


def _coerce_string(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, Decimal):
        if raw == raw.to_integral_value():
            return str(int(raw))
        return format(raw, "f").rstrip("0").rstrip(".")
    s = str(raw).strip()
    if re.fullmatch(r"\d+\.0+", s):
        return s.split(".", 1)[0]
    return s


def _build_sql(field_name: str, table_sql: str, updated_at_field: str, updated_since: str, limit: int) -> str:
    # 直接拼 SQL，便于日志排查和肉眼确认最终查询条件。
    # Doris 对 DISTINCT + ORDER BY 非聚合字段会报“should be grouped by”，因此这里只保留筛选条件。
    field_expr = f"TRIM(CAST(`{field_name}` AS CHAR))"
    sql = (
        f"SELECT DISTINCT {field_expr} AS n "
        f"FROM {table_sql} "
        f"WHERE `{updated_at_field}` >= '{updated_since}' "
        f"AND CHAR_LENGTH({field_expr}) BETWEEN %s AND %s "
        f"AND {field_expr} != ''"
    )
    if limit > 0:
        sql += f" LIMIT {int(limit)}"
    return sql


def _fetch_dictionary(field_name: str, env_table: str, log_prefix: str) -> List[str]:
    """从 Doris 拉取一列字符串，作为 AC 自动机的「模式串集合」。

    说明：AC 只关心「有哪些子串要在正文中找」；本函数负责连库、长度过滤、增量时间条件、
    分批 fetch、去重与类型规整（见 ``_coerce_string``）。拉取结果传给 ``build_automaton``。
    """
    host = os.environ.get("DORIS_HOST", "").strip()
    if not host:
        logger.warning("%s 未设置 DORIS_HOST，无法加载 AC 字典", log_prefix)
        return []

    table_sql = _qualified_table_sql(env_table)
    if not table_sql:
        logger.warning("%s 表名解析失败，检查 DORIS_DATABASE / %s", log_prefix, env_table)
        return []

    try:
        import pymysql
    except ImportError:
        logger.warning(
            "%s 未安装 pymysql。Python=%s 请执行: %s -m pip install pymysql",
            log_prefix,
            sys.executable,
            sys.executable,
        )
        return []

    port = int(os.environ.get("DORIS_PORT", "9030"))
    user = os.environ.get("DORIS_USER", "").strip()
    password = os.environ.get("DORIS_PASSWORD", "")
    session_db = os.environ.get("DORIS_SESSION_DATABASE", "").strip()

    if env_table == "DORIS_BUYER_NICKNAME_TABLE":
        # 买家昵称字典默认更严格，避免把整张大表直接灌进 AC。
        min_len = int(os.environ.get("DORIS_BUYER_NICKNAME_MIN_LEN", "2"))
        max_len = int(os.environ.get("DORIS_BUYER_NICKNAME_MAX_LEN", "32"))
        limit = int(os.environ.get("DORIS_BUYER_NICKNAME_LOAD_LIMIT", "0"))
        fetch = int(os.environ.get("DORIS_BUYER_NICKNAME_FETCH_SIZE", "10000"))
    else:
        min_len = int(os.environ.get("EXPRESS_AC_MIN_LEN", "1"))
        max_len = int(os.environ.get("EXPRESS_AC_MAX_LEN", "64"))
        limit = int(os.environ.get("EXPRESS_AC_LOAD_LIMIT", "0"))
        fetch = int(os.environ.get("EXPRESS_AC_FETCH_SIZE", "10000"))

    updated_at_field = os.environ.get("DORIS_AC_UPDATED_AT_FIELD", "updatedAt").strip()
    updated_since = os.environ.get("DORIS_AC_UPDATED_AT_SINCE", "2025-09-01").strip()
    sql = _build_sql(field_name, table_sql, updated_at_field, updated_since, limit)

    out: List[str] = []
    seen: Set[str] = set()

    conn = None
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=session_db or None,
            charset="utf8mb4",
            connect_timeout=int(os.environ.get("DORIS_CONNECT_TIMEOUT", "8")),
            read_timeout=int(os.environ.get("DORIS_READ_TIMEOUT", "120")),
        )
    except Exception as e:
        logger.warning("%s 连接 Doris 失败: %s", log_prefix, e, exc_info=True)
        return []

    logger.info(
        "%s 准备执行 Doris SQL, table=%s, field=%s, updated_at_field=%s, updated_since=%s, min_len=%s, max_len=%s, limit=%s",
        log_prefix,
        table_sql,
        field_name,
        updated_at_field,
        updated_since,
        min_len,
        max_len,
        limit,
    )
    logger.debug("%s Doris SQL: %s", log_prefix, sql)

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (min_len, max_len))
            while True:
                rows = cur.fetchmany(fetch)
                if not rows:
                    break
                for row in rows:
                    s = _coerce_string(row[0] if row else None)
                    if len(s) < min_len or not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
                    if os.environ.get("EXPRESS_AC_DEBUG", "").strip().lower() in ("1", "true", "yes"):
                        logger.info("%s 样例词: %r len=%d", log_prefix, s, len(s))
    except Exception as e:
        logger.warning("%s 字典 SQL 失败: %s", log_prefix, e, exc_info=True)
        return []
    finally:
        if conn is not None:
            conn.close()

    if not out:
        logger.warning(
            "%s 字典为 0 条（检查表数据、CHAR_LENGTH 范围 %s~%s、库名是否正确）",
            log_prefix,
            min_len,
            max_len,
        )
    else:
        logger.info("%s 字典加载 %d 条", log_prefix, len(out))
        if os.environ.get("EXPRESS_AC_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            logger.info("%s 前5条样例=%s", log_prefix, out[:5])
    return out


def fetch_express_dictionary() -> List[str]:
    return _fetch_dictionary("expressNumber", "DORIS_EXPRESS_TABLE", "[express_ac]")


def fetch_buyer_nickname_dictionary() -> List[str]:
    return _fetch_dictionary(
        "buyersNickname", "DORIS_BUYER_NICKNAME_TABLE", "[buyer_nickname_ac]"
    )


def build_automaton(words: List[str]) -> Any | None:
    """把词典串编译成 pyahocorasick 的 AC 自动机。

    参数 ``words`` 为 Doris 拉取并去重后的字符串列表（如运单号、昵称），可含数万条。

    实现细节（与库 API 的对应关系）
    ------------------------------
    - ``Automaton()``：创建空自动机，内部先建 trie 节点；此时还不能 ``iter`` 扫描。
    - ``add_word(word, value)``：向 trie 插入 ``word``；``value`` 为匹配成功时回调带回的**载荷**。
      这里载荷与词本身相同（``w``），便于 ``iter`` 直接拿到匹配到的字符串，无需再查表。
    - ``make_automaton()``：**必须调用**：在此阶段根据 trie 计算所有失配边与输出集合，
      之后自动机变为只读扫描结构；未调用则行为未定义。

    复杂度：设词典总字符数为 U、词数为 M，构建通常为 O(U) 量级（库实现细节以官方为准）；
    构建是一次性成本，服务进程内缓存复用（见 ``get_*_automaton``）。
    """
    if ahocorasick is None or not words:
        return None
    auto = ahocorasick.Automaton()
    for w in words:
        if w:
            # 第二个参数 w：匹配到该词时 iter 会把这个对象 yield 出来，这里即「命中的词典串」。
            auto.add_word(w, w)
    # 编译失配链与输出；此前仅注册了词，尚不能用于扫描。
    auto.make_automaton()
    return auto


def _resolve_spans(matches: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    """在 AC 原始命中列表上做去重与非重叠筛选，减少重复/嵌套标注。

    输入 ``matches`` 每项为 ``(start, end, word)``，其中 ``end`` 为**右开**切片上界，
    与 ``iter_matches`` 最终返回约定一致（子串为 ``text[start:end]``）。

    处理分三步：

    1. **同区间去重**（``uniq``）：
       若完全相同 ``(start, end)`` 出现多次（理论上 AC 对同一词不应重复报告同一终点，
       但防御性保留），保留 ``word`` **更长**的那条，避免短词覆盖长词语义。

    2. **非重叠贪心**（按长度优先）：
       将候选按 ``(-长度, start)`` 排序，即**更长的 span 优先、同长度则更靠左优先**。
       依次尝试加入 ``kept``：若与已接受的任一条在区间上相交（非「完全分离」），则丢弃。
       直观效果：在重叠的多个命中里，优先保留「更长」的匹配，减少碎片。

    3. **输出顺序**：按 ``start`` 再 ``end`` 升序排序，便于日志阅读与与正则结果合并。

    与纯 AC 输出的关系：AC 只负责「报告所有词典词在文本中的出现」；
    业务上若不想同一字符被多个标签叠盖，需要本函数或上游再做策略取舍。
    """
    if not matches:
        return []
    uniq: dict[Tuple[int, int], Tuple[int, int, str]] = {}
    for s, e, w in matches:
        k = (s, e)
        if k not in uniq or len(w) > len(uniq[k][2]):
            uniq[k] = (s, e, w)
    lst = list(uniq.values())
    # 先尝试更长的 span，使后续「不相交才保留」偏向长匹配。
    lst.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    kept: List[Tuple[int, int, str]] = []
    for m in lst:
        # 区间相交：非 (m 完全在 o 左侧) 且非 (m 完全在 o 右侧)。分离条件：m.end<=o.start 或 m.start>=o.end。
        if any(not (m[1] <= o[0] or m[0] >= o[1]) for o in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda x: (x[0], x[1]))


def _buyer_nickname_whitespace_boundary_ok(text: str, start: int, end: int) -> bool:
    """判断 AC 命中的 ``text[start:end]`` 是否像「与周围正文用空白隔开的昵称」。

    群聊里匿名买家昵称与前后其它内容之间应有空白（见业务约定）；否则 ``abcnd`` 这类无空白串里
    被 AC 扫出的单字 ``b`` 不应视为买家昵称。

    规则（``end`` 为半开右界）：
    - **左界**：``start == 0`` 或 ``text[start - 1]`` 为空白（``str.isspace()``，含空格/制表/换行等）。
    - **右界**：``end == len(text)`` 或 ``text[end]`` 为空白。

    词典串内部可含空格（如 ``ab cnd``），仅检查 span **外侧**紧邻字符；AC 整段命中后
    与原文一致，``ab cnd`` 在 ``… xx ab cnd yy …`` 中左右邻接空白则通过。
    """
    if start < 0 or end > len(text) or start >= end:
        return False
    if start > 0 and not text[start - 1].isspace():
        return False
    if end < len(text) and not text[end].isspace():
        return False
    return True


def iter_buyer_nickname_matches(text: str, auto: Any | None) -> List[Tuple[int, int, str]]:
    """对买家昵称专用 AC 扫描 ``text``，先做子串命中，再按空白边界过滤，最后 ``_resolve_spans``。

    与 ``iter_matches`` 的差异：快递面单等编码类字段允许紧贴数字字母；买家昵称在 AC 命中后
    必须满足 ``_buyer_nickname_whitespace_boundary_ok``，避免嵌入在无空白长串中的短词典误标。

    环境变量 ``BUYER_NICKNAME_WS_BOUNDARY`` 为 ``0``/``false``/``no`` 时关闭该过滤（调试用），
    行为与 ``iter_matches`` 一致。
    """
    if not text or auto is None:
        return []
    raw: List[Tuple[int, int, str]] = []
    for end_index, word in auto.iter(text):
        le = len(word)
        start = end_index - le + 1
        if start < 0:
            continue
        if text[start : end_index + 1] != word:
            continue
        raw.append((start, end_index + 1, word))

    if os.environ.get("BUYER_NICKNAME_WS_BOUNDARY", "1").strip().lower() in ("0", "false", "no"):
        filtered = raw
    else:
        filtered = [t for t in raw if _buyer_nickname_whitespace_boundary_ok(text, t[0], t[1])]

    return _resolve_spans(filtered)


def iter_matches(text: str, auto: Any | None) -> List[Tuple[int, int, str]]:
    """对整段 ``text`` 运行 AC 自动机，返回去重叠后的 ``(start, end, word)`` 列表。

    参数 ``auto`` 为 ``build_automaton`` 的返回值（``pyahocorasick.Automaton``），
    若为 ``None``（未安装库、词典为空、或环境关闭 AC）则返回空列表。

    ``pyahocorasick`` 的 ``iter`` 约定（与本函数坐标换算）
    ----------------------------------------------------
    ``for end_index, value in auto.iter(text)`` 中：

    - ``end_index``：**匹配最后一个字符在 text 中的下标**（闭区间端点），不是切片右界。
    - ``value``：即 ``add_word`` 时存入的载荷，本项目中等于匹配到的词 ``word``。

    因此若词长为 ``L``，词在 text 中占据的闭区间为
    ``[end_index - L + 1, end_index]``，转成 Python 半开切片为
    ``start = end_index - L + 1``，``end_exclusive = end_index + 1``。

    额外校验 ``text[start:end_exclusive] == word``：
    防止编码/Unicode 规范化等极端情况下下标与内容不一致（防御性；正常应成立）。

    返回值经 ``_resolve_spans`` 过滤，供 ``model.py`` 生成 Label Studio 的文本区域。
    """
    if not text or auto is None:
        return []
    raw: List[Tuple[int, int, str]] = []
    for end_index, word in auto.iter(text):
        le = len(word)
        start = end_index - le + 1
        if start < 0:
            continue
        if text[start : end_index + 1] != word:
            continue
        # end_index+1：与 Python slice 右开约定对齐，即匹配区间为 text[start:end_index+1]。
        raw.append((start, end_index + 1, word))
    return _resolve_spans(raw)


def get_express_automaton(force_reload: bool = False) -> Any | None:
    """返回「快递面单号 / 运单号」词典对应的进程内单例 AC 自动机。

    加载策略（与环境变量）
    ----------------------
    - ``EXPRESS_USE_AC``：若为 ``0``/``false``/``no``，直接返回 ``None``，上层仅用其它规则。
    - ``EXPRESS_AC_REFRESH_SECS``（默认 600）：成功构建后，在这么多秒内**不重复**拉 Doris、
      不重建自动机（除非 ``force_reload=True``）。0 表示每次调用都尝试刷新（慎用）。
    - ``EXPRESS_AC_EMPTY_RETRY_SECS``（默认 45）：若上次加载后自动机为 ``None``（词典空或
      构建失败），在这么多秒内**不再打 Doris**，避免故障时每个请求都触发重试风暴。

    并发：全程在 ``_load_lock`` 内读写全局单例，避免多线程同时构建两份大自动机。

    参数 ``force_reload``：容器管理或运维在词典更新后可设为 True 强制重新拉取；
    ``_wsgi`` 预热与手动刷新脚本也会用 True。
    """
    global _express_automaton, _last_success_monotonic, _last_attempt_monotonic

    if ahocorasick is None:
        logger.warning("[express_ac] 未安装 pyahocorasick，跳过 AC")
        return None

    if os.environ.get("EXPRESS_USE_AC", "1").strip().lower() in ("0", "false", "no"):
        return None

    refresh = max(0, int(os.environ.get("EXPRESS_AC_REFRESH_SECS", "600")))
    empty_retry = max(0, float(os.environ.get("EXPRESS_AC_EMPTY_RETRY_SECS", "45")))
    now = time.monotonic()

    with _load_lock:
        # 命中缓存窗口：已有自动机且未过期，直接返回（热路径）。
        if not force_reload and refresh > 0 and _express_automaton is not None:
            if _last_success_monotonic > 0 and (now - _last_success_monotonic) < refresh:
                return _express_automaton

        # 空结果退避：上次没建成机子，短时间内不再访问 Doris。
        if (
            not force_reload
            and empty_retry > 0
            and _express_automaton is None
            and _last_attempt_monotonic > 0
            and (now - _last_attempt_monotonic) < empty_retry
        ):
            return None

        words = fetch_express_dictionary()
        _express_automaton = build_automaton(words)
        _last_attempt_monotonic = now
        if _express_automaton is not None:
            _last_success_monotonic = now
        return _express_automaton


def get_buyer_nickname_automaton(force_reload: bool = False) -> Any | None:
    """返回「买家昵称」词典对应的进程内单例 AC 自动机。

    与 ``get_express_automaton`` 对称；环境变量前缀为 ``BUYER_NICKNAME_*``，
    词典字段与长度/条数限制见 ``_fetch_dictionary`` 中 ``DORIS_BUYER_NICKNAME_TABLE`` 分支。

    同样受共享的 ``_last_success_monotonic`` / ``_last_attempt_monotonic`` 影响（见模块顶部说明）。
    """
    global _buyer_nickname_automaton, _last_success_monotonic, _last_attempt_monotonic

    if ahocorasick is None:
        logger.warning("[buyer_nickname_ac] 未安装 pyahocorasick，跳过 AC")
        return None

    if os.environ.get("BUYER_NICKNAME_USE_AC", "1").strip().lower() in ("0", "false", "no"):
        return None

    refresh = max(0, int(os.environ.get("BUYER_NICKNAME_AC_REFRESH_SECS", "600")))
    empty_retry = max(0, float(os.environ.get("BUYER_NICKNAME_AC_EMPTY_RETRY_SECS", "45")))
    now = time.monotonic()

    with _load_lock:
        if not force_reload and refresh > 0 and _buyer_nickname_automaton is not None:
            if _last_success_monotonic > 0 and (now - _last_success_monotonic) < refresh:
                return _buyer_nickname_automaton

        if (
            not force_reload
            and empty_retry > 0
            and _buyer_nickname_automaton is None
            and _last_attempt_monotonic > 0
            and (now - _last_attempt_monotonic) < empty_retry
        ):
            return None

        words = fetch_buyer_nickname_dictionary()
        _buyer_nickname_automaton = build_automaton(words)
        _last_attempt_monotonic = now
        if _buyer_nickname_automaton is not None:
            _last_success_monotonic = now
        return _buyer_nickname_automaton


def preload_automata() -> None:
    """容器启动时预热两台 AC 自动机，避免首个 HTTP 预测请求才触发 Doris + 构建大自动机导致超时。

    由 ``_wsgi`` 在应用装载阶段调用；单条失败只记日志，不阻断另一台预热。
    """
    logger.info("[express_ac] 开始预热 AC 自动机")
    try:
        get_express_automaton(force_reload=True)
    except Exception:
        logger.exception("[express_ac] 预热快递面单 AC 失败")
    try:
        get_buyer_nickname_automaton(force_reload=True)
    except Exception:
        logger.exception("[express_ac] 预热买家昵称 AC 失败")
    logger.info("[express_ac] AC 自动机预热完成")
