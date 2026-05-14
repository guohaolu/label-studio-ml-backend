"""从 Doris 拉取字典，构建 Aho-Corasick 在正文中做子串匹配。"""
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

from express_doris import _qualified_table_sql

_automaton = None  # type: ignore[var-annotated]
_last_success_monotonic = 0.0
_last_attempt_monotonic = 0.0
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
    field_expr = f"TRIM(CAST(`{field_name}` AS CHAR))"
    sql = (
        f"SELECT DISTINCT {field_expr} AS n "
        f"FROM {table_sql} "
        f"WHERE `{updated_at_field}` >= '{updated_since}' "
        f"AND CHAR_LENGTH({field_expr}) BETWEEN %s AND %s "
        f"AND {field_expr} != '' "
    )
    if limit > 0:
        sql += f"LIMIT {int(limit)}"
    return sql


def _fetch_dictionary(field_name: str, env_table: str, log_prefix: str) -> List[str]:
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
        limit = int(os.environ.get("DORIS_BUYER_NICKNAME_LOAD_LIMIT", "200000"))
        fetch = int(os.environ.get("DORIS_BUYER_NICKNAME_FETCH_SIZE", "10000"))
    else:
        min_len = int(os.environ.get("EXPRESS_AC_MIN_LEN", "1"))
        max_len = int(os.environ.get("EXPRESS_AC_MAX_LEN", "64"))
        limit = int(os.environ.get("EXPRESS_AC_LOAD_LIMIT", "0"))
        fetch = int(os.environ.get("EXPRESS_AC_FETCH_SIZE", "10000"))

    # 直接把更新时间条件和排序写进 SQL，便于日志排查，也更容易读懂。
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


def build_automaton(words: List[str]):
    if ahocorasick is None or not words:
        return None
    auto = ahocorasick.Automaton()
    for w in words:
        if w:
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


def iter_matches(text: str, auto) -> List[Tuple[int, int, str]]:
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


def preload_automata() -> None:
    """容器启动时预热 AC 自动机，避免首个请求阻塞。"""
    logger.info("[express_ac] 开始预热 AC 自动机")
    try:
        get_express_automaton(force_reload=True)
    except Exception:
        logger.exception("[express_ac] 预热快递面单 AC 失败")
    try:
        buyer_words = fetch_buyer_nickname_dictionary()
        logger.info("[express_ac] 买家昵称预热字典条数=%d", len(buyer_words))
        build_automaton(buyer_words)
    except Exception:
        logger.exception("[express_ac] 预热买家昵称 AC 失败")
    logger.info("[express_ac] AC 自动机预热完成")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="AC 字典自检")
    parser.add_argument("--offline", action="store_true", help="离线样例验证")
    args = parser.parse_args()

    text = "买家昵称 张三丰，快递单号 92928429344"
    if args.offline:
        demo = build_automaton(["张三丰"])
        print("offline AC built:", demo is not None)
        print("matches:", iter_matches(text, demo))
