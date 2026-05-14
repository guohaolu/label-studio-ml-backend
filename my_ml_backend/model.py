"""
群聊文本 NER 预标注：正则标签 + AC 自动机标签。

当前支持：
- 手机号
- 发货计划单号
- 送装单号
- 合并单号
- 快递面单
- 买家昵称
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse, PredictionValue

from express_ac import (
    build_automaton,
    fetch_buyer_nickname_dictionary,
    fetch_express_dictionary,
    get_express_automaton,
    iter_matches,
)

logger = logging.getLogger(__name__)

_PHONE_LS_LABEL = "手机号"
_PLAN_LS_LABEL = "发货计划单号"
_SZD_LS_LABEL = "送装单号"
_MERGE_LS_LABEL = "合并单号"
_EXPRESS_LS_LABEL = "快递面单"
_BUYER_NICKNAME_LS_LABEL = "买家昵称"

_DATE_YMD_IN_ORDER = r"(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])"
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?:-\d{3,8}|转\d+)?(?!\d)")
_PLAN_RE = re.compile(rf"(?<![A-Za-z0-9])JH{_DATE_YMD_IN_ORDER}\d{{4,8}}(?!\d)")
_SZD_RE = re.compile(rf"(?<![A-Za-z0-9])SZD{_DATE_YMD_IN_ORDER}\d{{6,16}}(?!\d)")
_MERGE_RE = re.compile(rf"(?<![A-Za-z0-9])HB{_DATE_YMD_IN_ORDER}\d{{4,10}}(?!\d)")

_REGEX_TAG_RULES: List[Tuple[str, re.Pattern[str], float]] = [
    (_PHONE_LS_LABEL, _PHONE_RE, 0.95),
    (_PLAN_LS_LABEL, _PLAN_RE, 1.0),
    (_SZD_LS_LABEL, _SZD_RE, 1.0),
    (_MERGE_LS_LABEL, _MERGE_RE, 1.0),
]


def _iter_spans(text: str, pattern: re.Pattern[str]) -> List[Tuple[int, int]]:
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
    return {
        "id": str(uuid4())[:8],
        "from_name": from_name,
        "to_name": to_name,
        "type": "labels",
        "value": {"start": start, "end": end, "text": text[start:end], "labels": [label]},
        "score": score,
    }


class NewModel(LabelStudioMLBase):
    def setup(self) -> None:
        self.set("model_version", "0.0.2")
        self.set("express_automaton", get_express_automaton(force_reload=True))
        buyer_words = fetch_buyer_nickname_dictionary()
        self.set("buyer_nickname_automaton", build_automaton(buyer_words))

    def predict(
        self,
        tasks: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ModelResponse:
        try:
            from_name, to_name, value_key = self.label_interface.get_first_tag_occurence(
                "Labels", "Text"
            )
        except Exception:
            from_name, to_name, value_key = "label", "text", "text"

        model_version = self.get("model_version")
        express_auto = self.get("express_automaton")
        buyer_auto = self.get("buyer_nickname_automaton")

        predictions: List[PredictionValue] = []
        for task in tasks:
            raw = task.get("data", {}).get(value_key, "")
            text = self.preload_task_data(task, raw)
            if not isinstance(text, str):
                text = str(text) if text is not None else ""

            result: List[Dict[str, Any]] = []
            for ls_label, pattern, rule_score in _REGEX_TAG_RULES:
                for start, end in _iter_spans(text, pattern):
                    result.append(_make_label_region(from_name, to_name, ls_label, text, start, end, rule_score))

            for start, end, _word in iter_matches(text, express_auto):
                result.append(_make_label_region(from_name, to_name, _EXPRESS_LS_LABEL, text, start, end, 1.0))

            for start, end, _word in iter_matches(text, buyer_auto):
                result.append(_make_label_region(from_name, to_name, _BUYER_NICKNAME_LS_LABEL, text, start, end, 1.0))

            result.sort(key=lambda r: (r["value"]["start"], r["value"]["end"]))
            score = sum(r["score"] for r in result) / len(result) if result else 0.0
            predictions.append(PredictionValue(result=result, score=score, model_version=model_version))

        return ModelResponse(predictions=predictions, model_version=model_version)
