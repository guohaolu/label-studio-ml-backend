# my_ml_backend：群聊文本 NER 预标注服务

本目录为 **Label Studio ML Backend** 自定义实现：对任务中的文本做 **命名实体识别（NER）预标注**，产出带 `start` / `end` / `labels` 的预测结果，供 Label Studio 项目消费。

实现入口见 `model.py`（类 `NewModel`）、`express_ac.py`（AC 字典匹配）、`express_doris.py`（候选抽取与 Doris 校验）。

---

## 标签名与代码常量（必须与 Label Studio 配置一致）

| Label Studio `value`（标签名） | 代码中的常量（`model.py`） |
|-------------------------------|---------------------------|
| 手机号 | `_PHONE_LS_LABEL` |
| 发货计划单号 | `_PLAN_LS_LABEL` |
| 送装单号 | `_SZD_LS_LABEL` |
| 合并单号 | `_MERGE_LS_LABEL` |
| 快递面单 | `_EXPRESS_LS_LABEL` |
| 买家昵称 | `_BUYER_NICKNAME_LS_LABEL` |

## 快递面单标签

### 匹配策略（逐标签）

- **手段（AC）**：从 Doris 表 ``DORIS_EXPRESS_TABLE``（默认 `furniture_tms_busi__express_detail`）拉取 ``expressNumber`` 建 AC，正文子串命中后经 ``iter_matches`` / ``_resolve_spans`` 输出。
- **手段（加载清洗）**：每条 ``expressNumber`` 先 ``_coerce_string``（数值/小数等转字符串），再 **去掉除 `0-9`、`a-z`、`A-Z`、`-` 以外的所有字符**（等价于 ``str.replace(r'[^0-9a-zA-Z-]', '', regex=True)``）；仅当清洗后长度在 **[``EXPRESS_AC_CLEANED_MIN_LEN``, ``EXPRESS_AC_CLEANED_MAX_LEN``]**（默认 **7～30**）时才加入 AC 词典。原因：库内面单字段常夹标点、备注，直接灌 AC 噪声大。
- **Doris 侧粗筛**：SQL 仍用 ``EXPRESS_AC_MIN_LEN`` / ``EXPRESS_AC_MAX_LEN``（默认 1～64）与 ``CHAR_LENGTH`` 限制拉取量；最终入 AC 以清洗后长度为准。
- **数据来源 / 契约**：``DORIS_EXPRESS_TABLE``、字段 ``expressNumber``；连接变量同 ``express_ac._fetch_dictionary``。

### 环境变量（摘录）

- ``EXPRESS_AC_CLEANED_MIN_LEN`` / ``EXPRESS_AC_CLEANED_MAX_LEN``：清洗后面单号长度闭区间，默认 ``7`` / ``30``。
- ``DORIS_AC_UPDATED_AT_FIELD``：字典 SQL 里 ``WHERE`` 增量条件的时间列名，默认 ``updatedAt``；表字段名不同须配置。
- ``DORIS_AC_UPDATED_AT_SINCE``：只拉该日期（含）之后更新的行，默认 ``2025-09-01``。
- ``EXPRESS_AC_MIN_LEN`` / ``EXPRESS_AC_MAX_LEN`` / ``EXPRESS_AC_LOAD_LIMIT`` / ``EXPRESS_AC_REFRESH_SECS`` 等：见 ``express_ac.py``。

## 买家昵称标签

### 匹配策略（逐标签）

- **手段（AC）**：从 Doris 拉取 `buyersNickname`，构建与快递面单相同的 **Aho-Corasick 自动机**，在正文中做**子串**扫描（`iter_buyer_nickname_matches` 内与 `pyahocorasick` 的 `iter` 一致）。
- **手段（业务过滤）**：AC 给出的每条命中 ``(start, end)``（``end`` 半开）须满足 **左邻为串首或空白、右邻为串尾或空白**（``str.isspace()``）；用于剔除如原文 ``abcnd``、词典含 ``b`` 时那种**嵌入在无空白连续串内**的误命中；词典中含空格的整词（如 ``ab cnd``）在原文 ``… ab cnd …`` 中仍按**整段子串**命中，只要**外侧**紧贴空白或边界即可。
- **去重叠**：过滤后仍走 ``_resolve_spans``（长 span 优先、互不重叠贪心），与快递等一致。
- **关闭边界过滤（调试）**：环境变量 ``BUYER_NICKNAME_WS_BOUNDARY`` 设为 ``0``/``false``/``no`` 时，买家昵称行为与纯 ``iter_matches`` 相同。
- **数据来源 / 契约**：表名 ``DORIS_BUYER_NICKNAME_TABLE``（未设置时默认为 `furniture_tms_busi__delivery_order`）、字段 `buyersNickname`；``DORIS_DATABASE`` 或表名中带库名。其它 Doris 相关变量与 `express_ac._fetch_dictionary` 一致。
- **已知误报**：左右邻接为空白但仍是正常英文词的一部分（如两个词之间恰好只有一个空格）时，短词典仍可能命中；更细粒度需分词或词表。
- **已知漏报**：中文等**无空白**紧贴场景（如 ``你好张三``）若业务要求标出昵称，当前「外侧须空白」会漏标；与「昵称与周围有空格」的约定一致时可接受。

### 环境变量（摘录）

- ``BUYER_NICKNAME_USE_AC``：为 ``0``/``false``/``no`` 时关闭买家昵称 AC。
- ``BUYER_NICKNAME_WS_BOUNDARY``：默认开启空白边界过滤；见上。
- ``BUYER_NICKNAME_AC_REFRESH_SECS`` / ``BUYER_NICKNAME_AC_EMPTY_RETRY_SECS`` / ``DORIS_BUYER_NICKNAME_*``：见 ``express_ac.py`` 与 Doris 配置说明。
