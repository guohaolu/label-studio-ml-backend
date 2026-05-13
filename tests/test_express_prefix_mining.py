import pytest

from my_ml_backend.express_prefix_mining import ExpressSample, mine_express_prefixes


@pytest.fixture
def sample_rows():
    """构造一组覆盖典型场景的样本。

    这个 fixture 的目的不是模拟真实全量库，而是验证 `mine_express_prefixes`
    在下面几类情况中都能稳定工作：
    - 同一前缀的多条样本能够被聚合
    - 不同公司编码会分别形成自己的前缀统计
    - 非法/空值样本会被忽略
    """
    return [
        # 顺丰：同前缀、同模式，多条样本，应该被统计为高频前缀
        {"tracking_number": "SF1234567890", "company_code": "SF"},
        {"tracking_number": "sf1234567891", "company_code": "SF"},
        {"tracking_number": "SF1234567892", "company_code": "SF"},
        # 京东：另一个前缀，验证“不同公司编码不会互相干扰”
        ExpressSample(tracking_number="JD9876543210", company_code="JD"),
        ExpressSample(tracking_number="JD9876543211", company_code="JD"),
        # 圆通：前缀更长，验证长前缀也能正常提取
        {"tracking_number": "YT202401011234", "company_code": "YT"},
        {"tracking_number": "YT202401011235", "company_code": "YT"},
        # 脏数据：应被忽略，避免污染统计结果
        {"tracking_number": "", "company_code": "XX"},
        {"tracking_number": None, "company_code": "XX"},
        {"tracking_number": "***", "company_code": "XX"},
    ]


def test_mine_express_prefixes_detects_top_prefixes_and_company_mapping(sample_rows):
    """验证基础统计能力：

    1. 能正确区分有效样本和脏数据
    2. 能正确识别全局高频前缀
    3. 能正确建立“公司编码 -> 前缀”的聚合关系
    4. 能正确统计前缀长度分布
    """
    result = mine_express_prefixes(sample_rows, min_count=2)

    print(result)

    # 7 条有效样本，3 条脏数据被忽略。
    assert result["total"] == 7
    assert result["invalid"] == 3

    # 全局高频前缀：只要达到最小次数阈值，就应该进入结果集。
    assert "SF" in result["top_prefixes"]
    assert "JD" in result["top_prefixes"]
    assert "YT" in result["top_prefixes"]

    # 频次统计应该与样本数量一致。
    assert result["prefix_counter"]["SF"] == 3
    assert result["prefix_counter"]["JD"] == 2
    assert result["prefix_counter"]["YT"] == 2

    # 公司编码维度的前缀聚合：同一公司编码下的前缀应被汇总。
    assert result["company_top_prefixes"]["SF"][0]["prefix"] == "SF"
    assert result["company_top_prefixes"]["SF"][0]["count"] == 3
    assert result["company_top_prefixes"]["JD"][0]["prefix"] == "JD"
    assert result["company_top_prefixes"]["YT"][0]["prefix"] == "YT"

    # 前缀长度分布：用于后续做“前缀 + 长度”的启发式约束。
    assert result["prefix_length_counter"]["SF"][12] == 3
    assert result["prefix_length_counter"]["JD"][12] == 2
    assert result["prefix_length_counter"]["YT"][14] == 2


def test_mine_express_prefixes_builds_normalized_patterns(sample_rows):
    """验证模式归一化能力：

    同一前缀下，具体数字变化不同的单号，应该被归并成同一个模式。
    这样后续可以基于“模式频次”做更稳定的规则判断。
    """
    result = mine_express_prefixes(sample_rows, min_count=2)

    # 归一化模式应能把同一前缀的数字变化合并。
    assert "SF<NUM:10>" in result["top_patterns"]
    assert "JD<NUM:10>" in result["top_patterns"]
    assert "YT<NUM:12>" in result["top_patterns"]

    # 归一化模式的频次应该与原始样本数一致。
    assert result["pattern_counter"]["SF<NUM:10>"] == 3
    assert result["pattern_counter"]["JD<NUM:10>"] == 2
    assert result["pattern_counter"]["YT<NUM:12>"] == 2

    # 公司维度的模式聚合结果也要正确。
    assert result["company_top_patterns"]["SF"][0]["pattern"] == "SF<NUM:10>"
    assert result["company_top_patterns"]["SF"][0]["count"] == 3
