"""
群聊文本 NER 预标注：正则标签 + 快递面单（AC 或 Doris 回退）。

历史上各标签是按需逐个加的，容易出现「常量 / 正则 / 找 span / 组 result」四段重复。
本模块将「仅正则、且结构相同」的标签收拢到 _REGEX_TAG_RULES，便于对照 README 与 Label Studio 配置。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse, PredictionValue

from express_ac import get_express_automaton, iter_express_matches

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label Studio 中 Labels 的 value 必须与下列字符串一致（见 README 对照表）
# ---------------------------------------------------------------------------

# 手机号：与 <Label value="手机号"/> 一致
_PHONE_LS_LABEL = "手机号"

# 发货计划单号、送装单号、合并单号：前缀不同，日期段相同，故抽出公共子模式避免抄错一处、三处全错
_PLAN_LS_LABEL = "发货计划单号"
_SZD_LS_LABEL = "送装单号"
_MERGE_LS_LABEL = "合并单号"

# 快递面单：与 <Label value="快递面单"/> 一致；匹配逻辑由 express_ac 提供
_EXPRESS_LS_LABEL = "快递面单"

# YYYYMMDD：年份 19xx/20xx + 月 01-12 + 日 01-31（与业务单号中嵌入的日期格式一致）
_DATE_YMD_IN_ORDER = r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"

# 大陆手机：15988530256、18532139674转8615、18466687773-4338
# 原因：(?<!\d)(?!\d) 避免从更长数字串中切出一段误当手机号
_PHONE_RE = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?:-\d{3,8}|转\d+)?(?!\d)"
)

# 原因：(?<![A-Za-z0-9]) 避免 JH2026… 嵌在更长字母数字串中间被误匹配
_PLAN_RE = re.compile(
    rf"(?<![A-Za-z0-9])JH{_DATE_YMD_IN_ORDER}\d{{4,8}}(?!\d)"
)
_SZD_RE = re.compile(
    rf"(?<![A-Za-z0-9])SZD{_DATE_YMD_IN_ORDER}\d{{6,16}}(?!\d)"
)
_MERGE_RE = re.compile(
    rf"(?<![A-Za-z0-9])HB{_DATE_YMD_IN_ORDER}\d{{4,10}}(?!\d)"
)

# （标签名, 已编译正则, 该标签预测分）— 仅用于「整段正则扫描」类实体；快递面单单独处理
_REGEX_TAG_RULES: List[Tuple[str, re.Pattern[str], float]] = [
    (_PHONE_LS_LABEL, _PHONE_RE, 0.95),
    (_PLAN_LS_LABEL, _PLAN_RE, 1.0),
    (_SZD_LS_LABEL, _SZD_RE, 1.0),
    (_MERGE_LS_LABEL, _MERGE_RE, 1.0),
]


def _iter_spans(text: str, pattern: re.Pattern[str]) -> List[Tuple[int, int]]:
    """在 text 上跑 pattern，返回半开区间 [start, end) 列表。"""
    if not text:
        return []
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


def _make_label_region(
    from_name: str,
    to_name: str,
    label: str,
    text: str,
    start: int,
    end: int,
    score: float,
) -> Dict[str, Any]:
    """组装一条 Label Studio labels 区域预测（结构需与 SDK / LS 导出格式一致）。"""
    span_text = text[start:end]
    return {
        "id": str(uuid4())[:8],
        "from_name": from_name,
        "to_name": to_name,
        "type": "labels",
        "value": {
            "start": start,
            "end": end,
            "text": span_text,
            "labels": [label],
        },
        "score": score,
    }


class NewModel(LabelStudioMLBase):
    """Label Studio ML Backend：对任务文本做规则 + 库校验式 NER 预标注。"""

    def setup(self) -> None:
        """启动时写入模型版本号，并预加载快递 AC 自动机到内存。"""
        self.set("model_version", "0.0.1")
        self.set("express_automaton", get_express_automaton(force_reload=True))

    def predict(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        对 tasks 中文本字段做预测。

        原因：from_name / to_name / 文本键名来自标注界面 XML，必须通过 label_interface 解析，
        解析失败时使用常见默认值，避免整个请求失败。
        """
        try:
            from_name, to_name, value_key = self.label_interface.get_first_tag_occurence(
                "Labels", "Text"
            )
        except Exception:
            from_name, to_name, value_key = "label", "text", "text"

        predictions: List[PredictionValue] = []
        model_version = self.get("model_version")

        task_texts: List[str] = []
        for task in tasks:
            raw = task.get("data", {}).get(value_key, "")
            text = self.preload_task_data(task, raw)
            if not isinstance(text, str):
                text = str(text) if text is not None else ""
            task_texts.append(text)

        auto = self.get("express_automaton")
        if auto is None:
            logger.warning("[predict] 快递面单: AC 未初始化，返回空结果, tasks=%d", len(task_texts))
        else:
            logger.info("[predict] 快递面单: AC 模式, tasks=%d", len(task_texts))
        express_spans_by_task = [iter_express_matches(t, auto) for t in task_texts]

        for text, express_spans in zip(task_texts, express_spans_by_task):
            result: List[Dict[str, Any]] = []

            for ls_label, pattern, rule_score in _REGEX_TAG_RULES:
                for start, end in _iter_spans(text, pattern):
                    result.append(
                        _make_label_region(
                            from_name,
                            to_name,
                            ls_label,
                            text,
                            start,
                            end,
                            rule_score,
                        )
                    )

            for start, end, _exp in express_spans:
                result.append(
                    _make_label_region(
                        from_name,
                        to_name,
                        _EXPRESS_LS_LABEL,
                        text,
                        start,
                        end,
                        1.0,
                    )
                )

            result.sort(key=lambda r: (r["value"]["start"], r["value"]["end"]))

            score = (
                sum(r["score"] for r in result) / len(result) if result else 0.0
            )
            predictions.append(
                PredictionValue(
                    result=result,
                    score=score,
                    model_version=model_version,
                )
            )

        return ModelResponse(predictions=predictions, model_version=model_version)