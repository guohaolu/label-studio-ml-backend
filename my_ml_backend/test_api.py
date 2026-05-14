"""
本目录下 API 测试。安装依赖后执行：

    pip install -r requirements-test.txt
    pytest test_api.py -v

若需校验 Doris 命中的「快递面单」，请先导出 DORIS_HOST / DORIS_USER / DORIS_PASSWORD 等环境变量。
"""

import json
import os

import pytest

from model import NewModel
from my_ml_backend.express_doris import extract_express_candidates

SAMPLE_TEXT = (
    "AAGU5t2oAJmLtqP7Ur1R5bv9 河南 在途 客户想要这个椅子周六派送 "
    "辛苦核实下物流时间给客户对接预约下 百世快运 92928429344"
)

LABEL_CONFIG = """<View>
  <Labels name="label" toName="text">
    <Label value="SPU"/>
    <Label value="SKU"/>
    <Label value="买家昵称"/>
    <Label value="手机号"/>
    <Label value="合并单号"/>
    <Label value="Channel SPU"/>
    <Label value="Channel SKU"/>
    <Label value="发货计划单号"/>
    <Label value="送装单号"/>
    <Label value="快递面单"/>
  </Labels>
  <Text name="text" value="$text"/>
</View>"""


@pytest.fixture
def client():
    from _wsgi import init_app

    app = init_app(model_class=NewModel)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_predict(client):
    request = {
        "tasks": [{"data": {"text": SAMPLE_TEXT}}],
        "label_config": LABEL_CONFIG,
        "project": "1.0",
    }

    response = client.post(
        "/predict", data=json.dumps(request), content_type="application/json"
    )
    assert response.status_code == 200
    body = json.loads(response.data)

    print("response body:")
    print(body)

    assert "results" in body
    assert len(body["results"]) == 1

    pred = body["results"][0]
    assert "result" in pred
    assert isinstance(pred["result"], list)

    print(extract_express_candidates("AAGU5t2oAJmLtqP7Ur1R5bv9 河南 在途 客户想要这个椅子周六派送 辛苦核实下物流时间给客户对接预约下 百世快运 92928429344"))


    labels_spans = []
    for item in pred["result"]:
        if item.get("type") != "labels":
            continue
        val = item["value"]
        labels_spans.append((val.get("text", ""), val.get("labels", [])))

    # 无 Doris 时：本条样本文本可能没有命中库，允许 0 条快递面单；接口与结构必须正常
    if os.environ.get("DORIS_HOST", "").strip():
        assert any(
            txt == "92928429344" and "快递面单" in lbls
            for txt, lbls in labels_spans
        ), "已配置 DORIS_HOST 时，样本文本应命中 92928429344 -> 快递面单"


def test_iter_buyer_nickname_rejects_embedded_short_word() -> None:
    """无空白长串内 AC 子串命中须被空白边界过滤掉。"""
    pytest.importorskip("ahocorasick")
    from express_ac import build_automaton, iter_buyer_nickname_matches

    auto = build_automaton(["b", "abc"])
    assert auto is not None
    spans = iter_buyer_nickname_matches("abcnd", auto)
    assert spans == []


def test_iter_buyer_nickname_allows_dictionary_phrase_with_spaces() -> None:
    pytest.importorskip("ahocorasick")
    from express_ac import build_automaton, iter_buyer_nickname_matches

    auto = build_automaton(["ab cnd"])
    assert auto is not None
    spans = iter_buyer_nickname_matches("xx ab cnd yy", auto)
    assert len(spans) == 1
    assert spans[0][0:2] == (3, 9)
    assert spans[0][2] == "ab cnd"
