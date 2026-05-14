"""从文本提取业务标签候选，并在 Doris 中做存在性校验。"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import List, Set, Tuple

logger = logging.getLogger(__name__)

# 面单号常见形态：字母数字与连字符，长度约 8～40（在【】/（）内或独立出现）
_TOKEN_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{7,39})")

_BRACKET_RES = (
    re.compile(r"【([^】]{1,512})】"),
    re.compile(r"（([^）]{1,512})）"),
    re.compile(r"\(([^\)]{1,512})\)"),
    re.compile(r"\[([^\]]{1,512})\]"),
)

# 承运商/物流词后的「纯数字单号」（直接写在中文后，如：百世快运 92928429344）
_LOGISTICS_THEN_DIGITS = re.compile(
    r"(?:百世快运|百世物流|百世|顺丰速运|顺丰|中通快递|中通|圆通速递|圆通|申通快递|申通"
    r"|韵达快递|韵达|京东物流|京东|德邦快递|德邦|极兔速递|极兔|菜鸟|安能|跨越|快运"
    r"|快递单号|快递|物流|运单号|运单|面单|单号)"
    r"[\s\u3000:：\-—]*([0-9]{8,20})(?![0-9])"
)

# 独立出现的纯数字单号（非 11 位大陆手机号）；与 TOKEN 互补，避免漏抽
_PLAIN_DIGIT_RUN = re.compile(r"(?<![0-9])([0-9]{10,15})(?![0-9])")
_CN_MOBILE_11 = re.compile(r"^1[3-9]\d{9}$")


def _trim_span(text: str, start: int, end: int) -> Tuple[int, int, str]:
    raw = text[start:end]
    ls = start + (len(raw) - len(raw.lstrip()))
    le = end - (len(raw) - len(raw.rstrip()))
    if le <= ls:
        return start, end, text[start:end].strip()
    return ls, le, text[ls:le]

def extract_express_candidates(text: str) -> List[Tuple[int, int, str]]:
    """
    返回 (start, end, 面单字符串)：
    - 从【】、（）、()、[] 的内文中用 TOKEN 切候选；
    - 承运商/物流词后的 8～20 位纯数字（如「百世快运 92928429344」）；
    - 全文 TOKEN；再补 10～15 位纯数字段（排除 11 位 1[3-9] 手机号）；
    - 与括号内文区间有重叠的独立段跳过，避免与括号内重复。
    """
    if not text:
        return []

    inner_ranges: List[Tuple[int, int]] = []
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int, str]] = []

    def add(start: int, end: int) -> None:
        s, e, t = _trim_span(text, start, end)
        if len(t) < 8:
            return
        key = (s, e)
        if key in seen:
            return
        seen.add(key)
        out.append((s, e, t))

    for rx in _BRACKET_RES:
        for m in rx.finditer(text):
            inner_s, inner_e = m.start(1), m.end(1)
            inner_ranges.append((inner_s, inner_e))
            inner = text[inner_s:inner_e]
            for tm in _TOKEN_RE.finditer(inner):
                add(inner_s + tm.start(), inner_s + tm.end())

    def overlaps_bracket_inner(a: int, b: int) -> bool:
        return any(a < ie and b > is_ for is_, ie in inner_ranges)

    for m in _LOGISTICS_THEN_DIGITS.finditer(text):
        a, b = m.start(1), m.end(1)
        if overlaps_bracket_inner(a, b):
            continue
        add(a, b)

    for tm in _TOKEN_RE.finditer(text):
        a, b = tm.start(), tm.end()
        if overlaps_bracket_inner(a, b):
            continue
        add(a, b)

    for m in _PLAIN_DIGIT_RUN.finditer(text):
        num = m.group(1)
        if len(num) == 11 and _CN_MOBILE_11.match(num):
            continue
        a, b = m.start(1), m.end(1)
        if overlaps_bracket_inner(a, b):
            continue
        add(a, b)

    return out


def _qualified_table_sql(env_table: str) -> str | None:
    """返回 `db`.`table` 或 `table`，非法则 None。"""
    db = os.environ.get("DORIS_DATABASE", "").strip()
    default_table = "furniture_tms_busi__express_detail"
    if env_table == "DORIS_BUYER_NICKNAME_TABLE":
        default_table = "furniture_tms_busi__plan_sheet"
    tbl = os.environ.get(env_table, default_table).strip()
    if not tbl:
        return None
    if "." in tbl:
        parts = tbl.split(".", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            return None
        segs = parts
    elif db:
        segs = [db, tbl]
    else:
        segs = [tbl]
    for p in segs:
        if not re.fullmatch(r"[A-Za-z0-9_]+", p):
            logger.warning("Doris 表名/库名含非法字符: %r", p)
            return None
    return ".".join(f"`{p}`" for p in segs)


def express_number_norm_sql() -> str:
    """库中面单号统一成与正文可比的字符串（逗号取首段；纯数字时不变）。"""
    return "SUBSTRING_INDEX(TRIM(CAST(`expressNumber` AS CHAR)), ',', 1)"


def lookup_express_numbers(numbers: List[str]) -> Set[str]:
    """
    在 Doris 表 furniture_tms_busi__express_detail（可配置）中按 expressNumber 匹配：
    使用 SUBSTRING_INDEX(..., ',', 1) 与文本候选对齐，兼容库中存成「单号,后缀」或与长度等拼在同一字段的情况；
    数值型列会先 CAST 再比较。
    需环境变量：DORIS_HOST；DORIS_USER；DORIS_PASSWORD；可选 DORIS_PORT(默认9030)、
    DORIS_DATABASE、DORIS_EXPRESS_TABLE。
    """
    uniq = sorted({n.strip() for n in numbers if n and n.strip()})
    if not uniq:
        return set()

    host = os.environ.get("DORIS_HOST", "").strip()
    if not host:
        logger.debug("未设置 DORIS_HOST，跳过快递面单库校验")
        return set()

    table_sql = _qualified_table_sql("DORIS_EXPRESS_TABLE")
    if not table_sql:
        return set()

    try:
        import pymysql
    except ImportError:
        logger.warning(
            "当前解释器未安装 pymysql，无法连接 Doris。Python=%s 请执行: %s -m pip install pymysql",
            sys.executable,
            sys.executable,
        )
        return set()

    port = int(os.environ.get("DORIS_PORT", "9030"))
    user = os.environ.get("DORIS_USER", "").strip()
    password = os.environ.get("DORIS_PASSWORD", "")
    session_db = os.environ.get("DORIS_SESSION_DATABASE", "").strip()

    found: Set[str] = set()
    chunk_size = min(int(os.environ.get("DORIS_IN_CHUNK", "400")), 1000)

    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=session_db or None,
            charset="utf8mb4",
            connect_timeout=int(os.environ.get("DORIS_CONNECT_TIMEOUT", "8")),
            read_timeout=int(os.environ.get("DORIS_READ_TIMEOUT", "30")),
        )
    except Exception as e:
        logger.warning("连接 Doris 失败: %s", e)
        return set()

    try:
        with conn.cursor() as cur:
            for i in range(0, len(uniq), chunk_size):
                chunk = uniq[i : i + chunk_size]
                placeholders = ",".join(["%s"] * len(chunk))
                norm = express_number_norm_sql()
                sql = (
                    f"SELECT `expressNumber` FROM {table_sql} "
                    f"WHERE {norm} IN ({placeholders})"
                )
                cur.execute(sql, chunk)
                for row in cur.fetchall():
                    if row and row[0] is not None:
                        raw = str(row[0]).strip()
                        found.add(raw)
                        main = raw.split(",", 1)[0].strip()
                        if main:
                            found.add(main)
    except Exception as e:
        logger.warning("Doris 查询 expressNumber 失败: %s", e)
    finally:
        conn.close()

    return found
