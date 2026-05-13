"""从历史快递面单与公司编码中自动挖掘高频前缀。

这个模块用于离线统计，不直接参与在线预测。

设计目标：
- 你已经有历史快递面单号和快递公司编码
- 不想手工维护各公司前缀
- 希望从数据库样本中自动归纳：
  - 高频前缀
  - 前缀 + 长度模式
  - 公司编码 -> 常见前缀

使用方式：
1. 从数据库读取历史样本
2. 调用 `mine_express_prefixes()` 统计前缀
3. 调用 `export_prefix_dictionary()` 导出词典，给 AC 自动机或启发式规则使用

示例：
```python
from my_ml_backend.express_prefix_mining import load_samples_from_doris, mine_express_prefixes

rows = load_samples_from_doris()
result = mine_express_prefixes(rows, min_count=10)
print(result["top_prefixes"])
```
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_PREFIX_RE = re.compile(r"^[A-Za-z]+")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class ExpressSample:
    """一条历史样本。"""

    tracking_number: str
    company_code: str


def normalize_tracking_number(value: Any) -> str:
    """统一清洗单号文本。

    规则：
    - 去掉首尾空白
    - 去掉中间空格和全角空格
    - 统一大写

    示例：
    ```python
    normalize_tracking_number(" sf 1234567890 ")  # "SF1234567890"
    ```
    """
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace(" ", "").replace("\u3000", "")
    return text.upper()


def extract_prefix(tracking_number: str) -> str:
    """提取字母前缀。

    例如：
    - `SF1234567890` -> `SF`
    - `ZTO9876543210` -> `ZTO`
    - `1234567890` -> 空串
    """
    text = normalize_tracking_number(tracking_number)
    m = _PREFIX_RE.match(text)
    return m.group(0) if m else ""


def shape_of(text: str) -> str:
    """把字符串归一化成结构形状串。

    例如：
    - `SF1234567890` -> `AA9999999999`
    - `ZTO-123456` -> `AAA-999999`
    """
    text = normalize_tracking_number(text)
    out: List[str] = []
    for ch in text:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    return "".join(out)


def split_prefix_and_body(tracking_number: str) -> Tuple[str, str]:
    """拆分前缀和主体。"""
    text = normalize_tracking_number(tracking_number)
    prefix = extract_prefix(text)
    return prefix, text[len(prefix) :]


def normalize_pattern(tracking_number: str) -> str:
    """把单号归一化成模式字符串。

    例如：
    - `SF1234567890` -> `SF<NUM:10>`
    - `ZTO123456789012` -> `ZTO<NUM:12>`
    - `JDAB123456` -> `JDAB<AA999999>`
    """
    text = normalize_tracking_number(tracking_number)
    prefix, body = split_prefix_and_body(text)
    if not prefix:
        return shape_of(text)

    digit_count = sum(1 for ch in body if ch.isdigit())
    alpha_count = sum(1 for ch in body if ch.isalpha())

    if body and alpha_count == 0 and digit_count > 0:
        return f"{prefix}<NUM:{digit_count}>"
    return f"{prefix}<{shape_of(body)}>"


def mine_express_prefixes(
    rows: Sequence[ExpressSample | Dict[str, Any]],
    min_count: int = 5,
) -> Dict[str, Any]:
    """从历史样本里挖掘高频前缀和模式。

    参数：
    - `rows`: 样本列表，每条至少包含 `tracking_number` 和 `company_code`
    - `min_count`: 最小出现次数，低于该阈值的前缀/模式会被过滤

    返回值：
    - `company_prefix_counter`: 公司编码 -> 前缀计数器
    - `prefix_counter`: 全局前缀计数器
    - `prefix_length_counter`: 前缀 -> 长度分布
    - `pattern_counter`: 归一化模式计数器
    - `top_prefixes`: 高频前缀列表
    - `top_patterns`: 高频模式列表
    """
    prefix_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    prefix_length_counter: DefaultDict[str, Counter[int]] = defaultdict(Counter)
    company_prefix_counter: DefaultDict[str, Counter[str]] = defaultdict(Counter)
    company_pattern_counter: DefaultDict[str, Counter[str]] = defaultdict(Counter)

    total = 0
    invalid = 0

    for row in rows:
        if isinstance(row, dict):
            tracking_number = row.get("tracking_number", "")
            company_code = row.get("company_code", "")
        else:
            tracking_number = row.tracking_number
            company_code = row.company_code

        tracking_number = normalize_tracking_number(tracking_number)
        company_code = str(company_code).strip().upper()

        if not tracking_number:
            invalid += 1
            continue

        # 只保留字母数字主体，避免明显脏数据干扰统计。
        if not _ALNUM_RE.search(tracking_number):
            invalid += 1
            continue

        prefix = extract_prefix(tracking_number)
        pattern = normalize_pattern(tracking_number)

        total += 1
        if prefix:
            prefix_counter[prefix] += 1
            prefix_length_counter[prefix][len(tracking_number)] += 1
            if company_code:
                company_prefix_counter[company_code][prefix] += 1

        pattern_counter[pattern] += 1
        if company_code:
            company_pattern_counter[company_code][pattern] += 1

    top_prefixes = [p for p, c in prefix_counter.items() if c >= min_count]
    top_patterns = [p for p, c in pattern_counter.items() if c >= min_count]

    # 公司编码 -> 高频前缀
    company_top_prefixes: Dict[str, List[Dict[str, Any]]] = {}
    for company_code, counter in company_prefix_counter.items():
        company_top_prefixes[company_code] = [
            {"prefix": prefix, "count": count}
            for prefix, count in counter.most_common()
            if count >= min_count
        ]

    company_top_patterns: Dict[str, List[Dict[str, Any]]] = {}
    for company_code, counter in company_pattern_counter.items():
        company_top_patterns[company_code] = [
            {"pattern": pattern, "count": count}
            for pattern, count in counter.most_common()
            if count >= min_count
        ]

    return {
        "total": total,
        "invalid": invalid,
        "prefix_counter": prefix_counter,
        "pattern_counter": pattern_counter,
        "prefix_length_counter": prefix_length_counter,
        "company_prefix_counter": company_prefix_counter,
        "company_pattern_counter": company_pattern_counter,
        "top_prefixes": top_prefixes,
        "top_patterns": top_patterns,
        "company_top_prefixes": company_top_prefixes,
        "company_top_patterns": company_top_patterns,
    }


def export_prefix_dictionary(result: Dict[str, Any], output_path: str | Path) -> None:
    """把统计结果导出成 JSON，供 AC 自动机或规则引擎加载。"""
    payload = {
        "total": result.get("total", 0),
        "invalid": result.get("invalid", 0),
        "top_prefixes": result.get("top_prefixes", []),
        "top_patterns": result.get("top_patterns", []),
        "company_top_prefixes": result.get("company_top_prefixes", {}),
        "company_top_patterns": result.get("company_top_patterns", {}),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _qualified_table_sql() -> str | None:
    """返回 `db`.`table` 或 `table`，非法则 None。"""
    db = os.environ.get("DORIS_DATABASE", "").strip()
    tbl = os.environ.get("DORIS_EXPRESS_TABLE", "furniture_tms_busi__express_detail").strip()
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


def load_samples_from_doris(
    tracking_col: Optional[str] = None,
    company_col: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[ExpressSample]:
    """从 Doris 中读取样本。

    依赖环境变量：
    - `DORIS_HOST`
    - `DORIS_USER`
    - `DORIS_PASSWORD`
    - 可选 `DORIS_PORT`，默认 9030
    - 可选 `DORIS_DATABASE`
    - 可选 `DORIS_EXPRESS_TABLE`

    默认字段名：
    - `tracking_number`：快递面单号
    - `company_code`：快递公司编码
    """
    host = os.environ.get("DORIS_HOST", "").strip()
    if not host:
        logger.debug("未设置 DORIS_HOST，跳过 Doris 读取")
        return []

    table_sql = _qualified_table_sql()
    if not table_sql:
        return []

    try:
        import pymysql
    except ImportError:
        logger.warning(
            "当前解释器未安装 pymysql，无法连接 Doris。Python=%s 请执行: %s -m pip install pymysql",
            sys.executable,
            sys.executable,
        )
        return []

    port = int(os.environ.get("DORIS_PORT", "9030"))
    user = os.environ.get("DORIS_USER", "").strip()
    password = os.environ.get("DORIS_PASSWORD", "")
    session_db = os.environ.get("DORIS_SESSION_DATABASE", "").strip()

    tracking_col = tracking_col or os.environ.get("DORIS_TRACKING_COLUMN", "tracking_number")
    company_col = company_col or os.environ.get("DORIS_COMPANY_COLUMN", "company_code")

    if not re.fullmatch(r"[A-Za-z0-9_]+", tracking_col):
        raise ValueError(f"非法 tracking_col: {tracking_col!r}")
    if not re.fullmatch(r"[A-Za-z0-9_]+", company_col):
        raise ValueError(f"非法 company_col: {company_col!r}")

    sql = f"SELECT `{tracking_col}`, `{company_col}` FROM {table_sql}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"

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
    except Exception as exc:
        logger.warning("连接 Doris 失败: %s", exc)
        return []

    samples: List[ExpressSample] = []
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            for row in cur.fetchall():
                if not row:
                    continue
                tracking_number = row[0] if len(row) > 0 else ""
                company_code = row[1] if len(row) > 1 else ""
                samples.append(
                    ExpressSample(
                        tracking_number=str(tracking_number or ""),
                        company_code=str(company_code or ""),
                    )
                )
    except Exception as exc:
        logger.warning("Doris 查询样本失败: %s", exc)
    finally:
        conn.close()

    return samples


def build_prefix_dictionary_from_db(
    output_path: str | Path,
    min_count: int = 5,
    limit: Optional[int] = None,
    tracking_col: Optional[str] = None,
    company_col: Optional[str] = None,
) -> Dict[str, Any]:
    """一站式：从 Doris 读取样本、统计前缀、导出 JSON。"""
    samples = load_samples_from_doris(
        tracking_col=tracking_col,
        company_col=company_col,
        limit=limit,
    )
    result = mine_express_prefixes(samples, min_count=min_count)
    export_prefix_dictionary(result, output_path)
    return result
