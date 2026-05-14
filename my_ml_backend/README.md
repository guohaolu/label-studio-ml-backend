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

## 买家昵称标签

买家昵称标签使用和快递面单相同的 AC 自动机模式：

- 数据表：`furniture_tms_busi__plan_sheet`
- 字段：`buyersNickname`
- 环境变量：
  - `DORIS_BUYER_NICKNAME_TABLE=furniture_tms_busi__plan_sheet`
  - `DORIS_DATABASE` 或直接在表名里写库名

启动后会从 Doris 拉取 `buyersNickname` 字典，构建 AC 自动机，并在文本中做子串命中。
