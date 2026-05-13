import logging
import os
import re
from typing import List, Dict, Optional, Tuple
from uuid import uuid4

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from express_ac import get_express_automaton, iter_express_matches
from express_doris import extract_express_candidates, lookup_express_numbers

logger = logging.getLogger(__name__)

# 与 Label Studio 配置中 <Label value="手机号"/> 一致
_PHONE_LS_LABEL = "手机号"

# 大陆手机常见形态：15988530256、18532139674转8615、18466687773-4338
# (?<!\d)(?!\d) 避免从更长数字串中切出一段当手机号
_PHONE_RE = re.compile(
    r"(?<!\d)1[3-9]\d{9}(?:-\d{3,8}|转\d+)?(?!\d)"
)


def _find_phone_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    return [(m.start(), m.end()) for m in _PHONE_RE.finditer(text)]


# 与配置中 <Label value="发货计划单号"/> 一致
_PLAN_LS_LABEL = "发货计划单号"

# 发货计划单号：JH + 8 位日期 + 4～8 位流水号，例如 JH20260511000141
_PLAN_RE = re.compile(
    r"(?<![A-Za-z0-9])JH(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4,8}(?!\d)"
)


def _find_plan_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    return [(m.start(), m.end()) for m in _PLAN_RE.finditer(text)]


# 与配置中 <Label value="送装单号"/> 一致
_SZD_LS_LABEL = "送装单号"

# 送装单号：SZD + YYYYMMDD + 后缀数字，例如 SZD202604081513431745
_SZD_RE = re.compile(
    r"(?<![A-Za-z0-9])SZD(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{6,16}(?!\d)"
)


def _find_szd_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    return [(m.start(), m.end()) for m in _SZD_RE.finditer(text)]


# 与配置中 <Label value="合并单号"/> 一致
_MERGE_LS_LABEL = "合并单号"

# 合并单号：HB + YYYYMMDD + 流水号，例如 HB2026050500012281
_MERGE_RE = re.compile(
    r"(?<![A-Za-z0-9])HB(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{4,10}(?!\d)"
)


def _find_merge_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    return [(m.start(), m.end()) for m in _MERGE_RE.finditer(text)]


# 请在标注配置中增加 <Label value="快递面单"/>（或与下述字符串一致）
_EXPRESS_LS_LABEL = "快递面单"


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
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

        # 优先：Doris 字典 + AC；否则「候选 + Doris IN」
        force_ac = os.environ.get("EXPRESS_AC_FORCE_RELOAD", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        auto = get_express_automaton(force_reload=force_ac)
        if auto is not None:
            logger.info("[predict] 快递面单: AC 模式, tasks=%d", len(task_texts))
            express_spans_by_task = [iter_express_matches(t, auto) for t in task_texts]
        else:
            logger.info("[predict] 快递面单: 回退 Doris IN, tasks=%d", len(task_texts))
            per_task_cands = [extract_express_candidates(t) for t in task_texts]
            all_nums = [x for c in per_task_cands for _, __, x in c]
            found_express = lookup_express_numbers(all_nums)
            express_spans_by_task = [
                [(s, e, x) for s, e, x in c if x in found_express]
                for c in per_task_cands
            ]

        for text, express_spans in zip(task_texts, express_spans_by_task):
            result = []
            for start, end in _find_phone_spans(text):
                span_text = text[start:end]
                result.append(
                    {
                        "id": str(uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [_PHONE_LS_LABEL],
                        },
                        "score": 0.95,
                    }
                )
            for start, end in _find_plan_spans(text):
                span_text = text[start:end]
                result.append(
                    {
                        "id": str(uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [_PLAN_LS_LABEL],
                        },
                        "score": 1.0,
                    }
                )
            for start, end in _find_szd_spans(text):
                span_text = text[start:end]
                result.append(
                    {
                        "id": str(uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [_SZD_LS_LABEL],
                        },
                        "score": 1.0,
                    }
                )
            for start, end in _find_merge_spans(text):
                span_text = text[start:end]
                result.append(
                    {
                        "id": str(uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [_MERGE_LS_LABEL],
                        },
                        "score": 1.0,
                    }
                )
            for start, end, _exp in express_spans:
                span_text = text[start:end]
                result.append(
                    {
                        "id": str(uuid4())[:8],
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": start,
                            "end": end,
                            "text": span_text,
                            "labels": [_EXPRESS_LS_LABEL],
                        },
                        "score": 1.0,
                    }
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
    
    # def fit(self, event, data, **kwargs):
    #     """
    #     This method is called each time an annotation is created or updated
    #     You can run your logic here to update the model and persist it to the cache
    #     It is not recommended to perform long-running operations here, as it will block the main thread
    #     Instead, consider running a separate process or a thread (like RQ worker) to perform the training
    #     :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
    #     :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
    #     """
    #
    #     # use cache to retrieve the data from the previous fit() runs
    #     old_data = self.get('my_data')
    #     old_model_version = self.get('model_version')
    #     print(f'Old data: {old_data}')
    #     print(f'Old model version: {old_model_version}')
    #
    #     # store new data to the cache
    #     self.set('my_data', 'my_new_data_value')
    #     self.set('model_version', 'my_new_model_version')
    #     print(f'New data: {self.get("my_data")}')
    #     print(f'New model version: {self.get("model_version")}')
    #
    #     print('fit() completed successfully.')