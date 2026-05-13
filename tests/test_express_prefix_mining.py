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
        # 顺丰：字母前缀 + 数字，来自常见文章里的“SF+数字”规则。
        {"tracking_number": "SF1234567890", "company_code": "SF"},
        {"tracking_number": "sf1234567891", "company_code": "SF"},
        {"tracking_number": "SF1234567892", "company_code": "SF"},
        # 京东：另一个字母前缀，验证“不同公司编码不会互相干扰”。
        ExpressSample(tracking_number="JD9876543210", company_code="JD"),
        ExpressSample(tracking_number="JD9876543211", company_code="JD"),
        # 圆通：前缀更长，验证长前缀也能正常提取。
        {"tracking_number": "YT202401011234", "company_code": "YT"},
        {"tracking_number": "YT202401011235", "company_code": "YT"},
        # 纯数字单号：来自文章里的“纯数字规则”，用于验证无字母前缀场景也能统计前缀候选。
        # 这里也验证“数字也可以是前缀”这一点，例如 03、77 这类前缀。
        {"tracking_number": "031234567890123", "company_code": "YD"},
        {"tracking_number": "031234567890124", "company_code": "YD"},
        {"tracking_number": "771234567890123", "company_code": "ST"},
        {"tracking_number": "771234567890124", "company_code": "ST"},
        # 脏数据：应被忽略，避免污染统计结果。
        {"tracking_number": "", "company_code": "XX"},
        {"tracking_number": None, "company_code": "XX"},
        {"tracking_number": "***", "company_code": "XX"},
    ]


def test_mine_express_prefixes_detects_top_prefixes_and_company_mapping(sample_rows):
    """验证基础统计能力。

    这个测试刻意混合了三类样本：
    1. 字母前缀 + 数字单号（例如 SF、JD、YT）
    2. 纯数字单号（模拟文章里的 0/3/7/55/77/9 开头规则，数字本身也可作为前缀）
    3. 脏数据

    目标是确认：
    - 有效样本会被正确统计
    - 纯数字单号即使没有字母前缀，也不会把统计流程弄坏
    - 公司编码维度和长度分布都能正常聚合
    """
    result = mine_express_prefixes(sample_rows, min_count=2)

    import json

    print(
        json.dumps(
            {
                "total": result["total"],
                "invalid": result["invalid"],
                "top_prefixes": result["top_prefixes"],
                "top_patterns": result["top_patterns"],
                "company_top_prefixes": result["company_top_prefixes"],
                "company_top_patterns": result["company_top_patterns"],
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    # 11 条有效样本，3 条脏数据被忽略。
    assert result["total"] == 11
    assert result["invalid"] == 3

    # 全局高频前缀：字母前缀和数字前缀都应该能被发现。
    assert "SF" in result["top_prefixes"]
    assert "JD" in result["top_prefixes"]
    assert "YT" in result["top_prefixes"]
    assert "77" in result["top_prefixes"]
    assert "03" in result["top_prefixes"]

    # 频次统计应该与样本数量一致。
    assert result["prefix_counter"]["SF"] == 3
    assert result["prefix_counter"]["JD"] == 2
    assert result["prefix_counter"]["YT"] == 2
    assert result["prefix_counter"]["77"] == 2
    assert result["prefix_counter"]["03"] == 2

    # 公司编码维度的前缀聚合：同一公司编码下的前缀应被汇总。
    assert result["company_top_prefixes"]["SF"][0]["prefix"] == "SF"
    assert result["company_top_prefixes"]["SF"][0]["count"] == 3
    assert result["company_top_prefixes"]["JD"][0]["prefix"] == "JD"
    assert result["company_top_prefixes"]["YT"][0]["prefix"] == "YT"
    assert result["company_top_prefixes"]["YD"][0]["prefix"] == "03"
    assert result["company_top_prefixes"]["ST"][0]["prefix"] == "77"

    # 前缀长度分布：用于后续做“前缀 + 长度”的启发式约束。
    assert result["prefix_length_counter"]["SF"][12] == 3
    assert result["prefix_length_counter"]["JD"][12] == 2
    assert result["prefix_length_counter"]["YT"][14] == 2
    assert result["prefix_length_counter"]["77"][15] == 2
    assert result["prefix_length_counter"]["03"][15] == 2


def test_mine_express_prefixes_builds_normalized_patterns(sample_rows):
    """验证模式归一化能力。

    这里额外加入纯数字单号，目的是确认：
    - 没有字母前缀的单号，会走纯数字模式统计
    - 有字母前缀的单号，会走 `前缀<NUM:n>` 这种归一化模式
    
    这和文章里的经验规则是一致的：有些快递本来就是纯数字开头，不能只依赖前缀。
    """
    result = mine_express_prefixes(sample_rows, min_count=2)

    # 归一化模式应能把同一前缀的数字变化合并。
    assert "SF<NUM:10>" in result["top_patterns"]
    assert "JD<NUM:10>" in result["top_patterns"]
    assert "YT<NUM:12>" in result["top_patterns"]
    assert "<NUM:15>" in result["top_patterns"]  # 纯数字样本也应该形成自己的模式。

    # 归一化模式的频次应该与原始样本数一致。
    assert result["pattern_counter"]["SF<NUM:10>"] == 3
    assert result["pattern_counter"]["JD<NUM:10>"] == 2
    assert result["pattern_counter"]["YT<NUM:12>"] == 2
    assert result["pattern_counter"]["<NUM:15>"] == 2

    # 公司维度的模式聚合结果也要正确。
    assert result["company_top_patterns"]["SF"][0]["pattern"] == "SF<NUM:10>"
    assert result["company_top_patterns"]["SF"][0]["count"] == 3
    assert result["company_top_patterns"]["YD"][0]["pattern"] == "<NUM:15>"
    assert result["company_top_patterns"]["YD"][0]["count"] == 2
    assert result["company_top_patterns"]["ST"][0]["pattern"] == "<NUM:15>"
    assert result["company_top_patterns"]["ST"][0]["count"] == 2
