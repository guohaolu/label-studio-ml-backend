"""从 Doris 拉取快递单号字典，构建 Aho-Corasick 在正文中做子串匹配。"""
from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from decimal import Decimal
from typing import List, Set, Tuple

logger = logging.getLogger("express_ac")

try:
    import ahocorasick
except ImportError:  # pragma: no cover
    ahocorasick = None

from express_doris import _qualified_table_sql, express_number_norm_sql

_automaton = None  # type: ignore[var-annotated]
_last_success_monotonic = 0.0
_last_attempt_monotonic = 0.0
_load_lock = threading.Lock()


def _coerce_waybill_string(raw: object) -> str:
    """Doris/pymysql 可能给出 Decimal；转成与正文一致的数字串，去掉无意义的 .0。"""
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


def fetch_express_dictionary() -> List[str]:
    """
    拉取 DISTINCT 主单号；与 lookup_express_numbers 使用同一套 SQL 归一化。
    expressNumber 为数值型时，避免对裸列用 LENGTH() 导致 Doris 侧异常或筛空。
    """
    host = os.environ.get("DORIS_HOST", "").strip()
    if not host:
        logger.warning("[express_ac] 未设置 DORIS_HOST，无法加载 AC 字典")
        return []

    table_sql = _qualified_table_sql()
    if not table_sql:
        logger.warning("[express_ac] 表名解析失败，检查 DORIS_DATABASE / DORIS_EXPRESS_TABLE")
        return []

    try:
        import pymysql
    except ImportError:
        logger.warning(
            "[express_ac] 未安装 pymysql。Python=%s 请执行: %s -m pip install pymysql",
            sys.executable,
            sys.executable,
        )
        return []

    port = int(os.environ.get("DORIS_PORT", "9030"))
    user = os.environ.get("DORIS_USER", "").strip()
    password = os.environ.get("DORIS_PASSWORD", "")
    session_db = os.environ.get("DORIS_SESSION_DATABASE", "").strip()

    norm = express_number_norm_sql()
    min_len = int(os.environ.get("EXPRESS_AC_MIN_LEN", "8"))
    max_len = int(os.environ.get("EXPRESS_AC_MAX_LEN", "64"))
    limit = int(os.environ.get("EXPRESS_AC_LOAD_LIMIT", "0"))

    inner = f"SELECT {norm} AS n FROM {table_sql}"
    sql = (
        "SELECT DISTINCT TRIM(t.n) AS n FROM ("
        + inner
        + ") t WHERE CHAR_LENGTH(TRIM(t.n)) BETWEEN %s AND %s AND TRIM(t.n) != ''"
    )
    if limit > 0:
        sql += f" LIMIT {int(limit)}"

    out: List[str] = []
    seen: Set[str] = set()
    fetch = int(os.environ.get("EXPRESS_AC_FETCH_SIZE", "10000"))

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
        logger.warning("[express_ac] 连接 Doris 失败: %s", e, exc_info=True)
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(sql, (min_len, max_len))
            while True:
                rows = cur.fetchmany(fetch)
                if not rows:
                    break
                for row in rows:
                    s = _coerce_waybill_string(row[0] if row else None)
                    if len(s) < min_len or not s or s in seen:
                        continue
                    seen.add(s)
                    out.append(s)
    except Exception as e:
        logger.warning("[express_ac] 字典 SQL 失败: %s", e, exc_info=True)
        return []
    finally:
        if conn is not None:
            conn.close()

    if not out:
        logger.warning(
            "[express_ac] 字典为 0 条（检查表数据、CHAR_LENGTH 范围 %s~%s、库名是否正确）",
            min_len,
            max_len,
        )
    else:
        logger.info("[express_ac] 字典加载 %d 条", len(out))
        if os.environ.get("EXPRESS_AC_DEBUG", "").strip().lower() in ("1", "true", "yes"):
            hit = "92928429344"
            logger.info(
                "[express_ac] 样例: 前3条=%s；含 %s ? %s",
                out[:3],
                hit,
                any(x == hit for x in out),
            )

    return out


def build_automaton(words: List[str]):
    if ahocorasick is None or not words:
        return None
    auto = ahocorasick.Automaton()
    for w in words:
        if not w:
            continue
        auto.add_word(w, w)
    auto.make_automaton()
    return auto


def _resolve_spans(matches: List[Tuple[int, int, str]]) -> List[Tuple[int, int, str]]:
    if not matches:
        return []
    uniq: dict[Tuple[int, int], Tuple[int, int, str]] = {}
    for s, e, w in matches:
        k = (s, e)
        if k not in uniq or len(w) > len(uniq[k][2]):
            uniq[k] = (s, e, w)
    lst = list(uniq.values())
    lst.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    kept: List[Tuple[int, int, str]] = []
    for m in lst:
        if any(not (m[1] <= o[0] or m[0] >= o[1]) for o in kept):
            continue
        kept.append(m)
    return sorted(kept, key=lambda x: (x[0], x[1]))


def iter_express_matches(text: str, auto) -> List[Tuple[int, int, str]]:
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
    return _resolve_spans(raw)


def get_express_automaton(force_reload: bool = False):
    global _automaton, _last_success_monotonic, _last_attempt_monotonic

    if ahocorasick is None:
        logger.warning("[express_ac] 未安装 pyahocorasick，跳过 AC")
        return None

    if os.environ.get("EXPRESS_USE_AC", "1").strip().lower() in ("0", "false", "no"):
        return None

    refresh = max(0, int(os.environ.get("EXPRESS_AC_REFRESH_SECS", "600")))
    empty_retry = max(0, float(os.environ.get("EXPRESS_AC_EMPTY_RETRY_SECS", "45")))
    now = time.monotonic()

    with _load_lock:
        if not force_reload and refresh > 0 and _automaton is not None:
            if _last_success_monotonic > 0 and (now - _last_success_monotonic) < refresh:
                return _automaton

        if (
            not force_reload
            and empty_retry > 0
            and _automaton is None
            and _last_attempt_monotonic > 0
            and (now - _last_attempt_monotonic) < empty_retry
        ):
            return None

        words = fetch_express_dictionary()
        _automaton = build_automaton(words)
        _last_attempt_monotonic = now
        if _automaton is not None:
            _last_success_monotonic = now
        return _automaton


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    parser = argparse.ArgumentParser(description="快递 AC 字典自检")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="不连 Doris，仅用内置样例单号验证 AC 与文本匹配",
    )
    args = parser.parse_args()

    text = (
        "AAGU5t2oAJmLtqP7Ur1R5bv9 河南 在途 客户想要这个椅子周六派送 "
        "辛苦核实下物流时间给客户对接预约下 百世快运 92928429344"
    )

    if args.offline:
        if ahocorasick is None:
            print("请先安装: pip install pyahocorasick")
            raise SystemExit(1)
        demo = build_automaton(["92928429344"])
        print("offline AC built:", demo is not None)
        print("matches:", iter_express_matches(text, demo))
        raise SystemExit(0)

    if not os.environ.get("DORIS_HOST", "").strip():
        print(
            "未设置 DORIS_HOST，无法从 Doris 拉字典。\n"
            "  PowerShell 示例：\n"
            '    $env:DORIS_HOST="你的FE地址"\n'
            '    $env:DORIS_PORT="9030"\n'
            '    $env:DORIS_USER="..."\n'
            '    $env:DORIS_PASSWORD="..."\n'
            '    $env:DORIS_SESSION_DATABASE="你的库名"   # 或 DORIS_DATABASE + 表名\n'
            "  然后： python express_ac.py\n"
            "  或先做离线验证（不连库）： python express_ac.py --offline"
        )
        raise SystemExit(1)

    a = get_express_automaton(force_reload=True)
    print("AC built:", a is not None)
    if a is not None:
        print("matches:", iter_express_matches(text, a))
