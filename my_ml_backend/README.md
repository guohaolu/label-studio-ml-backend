# my_ml_backend：群聊文本 NER 预标注服务

本目录为 **Label Studio ML Backend** 自定义实现：对任务中的文本做 **命名实体识别（NER）预标注**，产出带 `start` / `end` / `labels` 的预测结果，供 Label Studio 项目消费。

实现入口见 `model.py`（类 `NewModel`）、`express_ac.py`（快递面单：Aho-Corasick 字典匹配）、`express_doris.py`（快递面单：候选抽取与 Doris 校验）。

---

## 如何运行

### 前置条件

- 已安装 [Docker](https://docs.docker.com/get-docker/) 与 Docker Compose（推荐），或本地 Python 3.12+（与 `Dockerfile` 中版本一致即可）。
- Label Studio 中本项目的标注配置里，**Labels 控件的 `value` 必须与下文「标签名与代码常量」表一致**，且存在与 `Text` 关联的 **Labels → Text** 配置（代码通过 `get_first_tag_occurence("Labels", "Text")` 解析 `from_name` / `to_name` / 文本字段名）。

### 方式一：Docker Compose（推荐）

在 **`my_ml_backend` 目录** 下执行：

```bash
cd my_ml_backend
docker-compose up --build
```

默认将服务暴露在 **`http://localhost:9090`**。健康检查示例：

```bash
curl http://localhost:9090/
```

期望返回类似：`{"status":"UP"}`。

在 Label Studio：**项目 → Settings → Machine Learning → Add Model**，填写后端地址 `http://localhost:9090`（若 Label Studio 在 Docker 内，勿用 `localhost` 指宿主机，需用宿主机 IP 或 Compose 网络内主机名，参见官方文档）。

**快递面单**依赖 Doris 时，在 `docker-compose.yml` 的 `environment` 中填写 `DORIS_HOST`、`DORIS_USER`、`DORIS_PASSWORD` 等（见下文「环境变量」）。不配 Doris 时，快递面单分支可能无字典或校验失败，其它 **纯正则** 标签仍可工作。

### 方式二：本地 Python（不经过 Docker）

在**仓库根目录**安装可编辑的 `label_studio_ml`，再安装本目录依赖并启动：

```bash
cd label-studio-ml-backend
pip install -e .
pip install -r my_ml_backend/requirements.txt
cd my_ml_backend
python _wsgi.py --host 0.0.0.0 -p 9090
```

生产式启动（与镜像内类似，多进程由 gunicorn 负责）：

```bash
cd my_ml_backend
gunicorn --bind :9090 --workers 1 --threads 8 --timeout 0 _wsgi:app
```

**Windows PowerShell** 设置 Doris 示例（按需修改值）：

```powershell
$env:DORIS_HOST = "你的 Doris FE 地址"
$env:DORIS_PORT = "9030"
$env:DORIS_USER = "用户名"
$env:DORIS_PASSWORD = "密码"
$env:DORIS_DATABASE = "库名"
$env:DORIS_EXPRESS_TABLE = "furniture_tms_busi__express_detail"
python _wsgi.py -p 9090
```

### 与 Label Studio 交换媒体/任务数据（可选）

若需从 Label Studio 拉取上传文件等，设置 `LABEL_STUDIO_URL`、`LABEL_STUDIO_API_KEY`；容器内访问宿主机 Label Studio 时**不要**使用 `localhost`，需使用宿主机在 Docker 网络中的可达地址。

---

## 环境变量（与「快递面单」及 Doris 相关）

| 变量 | 含义 |
|------|------|
| `DORIS_HOST` | Doris FE 地址；不设置则无法连库拉字典或做 IN 校验。 |
| `DORIS_PORT` | 默认 `9030`（MySQL 协议查询端口）。 |
| `DORIS_USER` / `DORIS_PASSWORD` | 连接凭据。 |
| `DORIS_DATABASE` | 库名；若 `DORIS_EXPRESS_TABLE` 已写成 `db.table` 则可不填。 |
| `DORIS_EXPRESS_TABLE` | 面单表，默认 `furniture_tms_busi__express_detail`；可为 `` `库`.`表` `` 形式。 |
| `DORIS_SESSION_DATABASE` | 连接后 `USE` 的库（可选）。 |
| `EXPRESS_USE_AC` | `1`（默认）尝试用 AC；`0` / `false` / `no` 则不走 AC，预测时走「候选 + Doris IN」回退。 |
| `EXPRESS_AC_REFRESH_SECS` | 字典刷新间隔（秒），默认 `600`；`0` 表示每次预测都尝试重拉字典（**慎用**，压力大）。 |
| `EXPRESS_AC_FORCE_RELOAD` | 设为 `1` / `true` / `yes` 时，单次预测强制重载字典（见 `model.py`）。 |
| `EXPRESS_AC_MIN_LEN` / `EXPRESS_AC_MAX_LEN` | 参与建 AC 字典的单号字符串长度范围，默认 `8`～`64`。 |
| `EXPRESS_AC_LOAD_LIMIT` | 字典条数上限，`0` 表示不限制。 |
| `DORIS_IN_CHUNK` | Doris `IN` 查询分批大小，默认 `400`。 |

快递字典自检脚本（需 Doris 或 `--offline`）：

```bash
cd my_ml_backend
python express_ac.py --offline
```

---

## 标签名与代码常量（必须与 Label Studio 配置一致）

| Label Studio `value`（标签名） | 代码中的常量（`model.py`） |
|-------------------------------|---------------------------|
| 手机号 | `_PHONE_LS_LABEL` |
| 发货计划单号 | `_PLAN_LS_LABEL` |
| 送装单号 | `_SZD_LS_LABEL` |
| 合并单号 | `_MERGE_LS_LABEL` |
| 快递面单 | `_EXPRESS_LS_LABEL` |

---

## 每个标签的匹配策略（与实现一一对应）

以下说明与 **`model.py` / `express_doris.py` / `express_ac.py` 当前逻辑**一致；若你修改了正则或快递流程，请同步更新本节。

### 1. 手机号（`手机号`）

- **匹配办法**：**单条 Python 正则**（`model.py` 中 `_PHONE_RE`），对整段任务文本做 `finditer`，每个匹配对应一个实体跨度 `[start, end)`。
- **模式要点**：
  - 主体为大陆常见 **11 位号段**：`1[3-9]` + 9 位数字（共 11 位）。
  - **前后不能是数字**：使用 `(?<!\d)` 与 `(?!\d)`，避免从更长数字串里「抠出」一段误当手机号。
  - **可选后缀**：`-` 后接 3～8 位数字，或 `转` 后接若干数字（用于分机/转写类形态，与业务中出现的字符串一致即可匹配）。
- **置信度**：预测里 `score` 为 **0.95**（其余规则类标签多为 1.0）。
- **与其它标签关系**：快递候选抽取里会 **排除** 符合大陆 11 位手机号的纯数字段，避免与「快递面单」数字候选冲突（见 `express_doris.py` 中 `_PLAIN_DIGIT_RUN` 与 `_CN_MOBILE_11`）。

### 2. 发货计划单号（`发货计划单号`）

- **匹配办法**：**单条 Python 正则**（`_PLAN_RE`），全文本扫描。
- **模式要点**：
  - 前缀 **`JH`**，且 **左侧不能是字母或数字**（`(?<![A-Za-z0-9])`），避免嵌入更长编码时被误切。
  - 日期部分：**`19` 或 `20` 开头的年份** + **月**（`01`～`12`）+ **日**（`01`～`31`），即与代码中 `(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])` 一致。
  - 流水号：**4～8 位数字**（`\d{4,8}`）。
  - **右侧不能再紧跟数字**（`(?!\d)`）。
- **示例形态**：`JH20260511000141`（`JH` + `20260511` + 流水）。

### 3. 送装单号（`送装单号`）

- **匹配办法**：**单条 Python 正则**（`_SZD_RE`），全文本扫描。
- **模式要点**：
  - 前缀 **`SZD`**，左侧同样要求非字母数字边界。
  - 日期：与上类似，**`19`/`20` 世纪年份 + 月日**（同上段月日子模式）。
  - 后缀数字：**6～16 位**（`\d{6,16}`），右侧不能紧跟数字。
- **示例形态**：`SZD202604081513431745`。

### 4. 合并单号（`合并单号`）

- **匹配办法**：**单条 Python 正则**（`_MERGE_RE`），全文本扫描。
- **模式要点**：
  - 前缀 **`HB`**，左侧非字母数字边界。
  - 日期：同上 **年月日** 子模式。
  - 流水号：**4～10 位数字**，右侧不能紧跟数字。
- **示例形态**：`HB2026050500012281`。

### 5. 快递面单（`快递面单`）

该标签是 **「文本中出现子串 + 业务库校验」** 类实体，**不是**一条固定正则写完所有面单格式。整体分 **主路径（AC）** 与 **回退路径（候选 + Doris IN）**，由 `model.py` 的 `predict` 决定。

#### 5.1 主路径：Doris 字典 + Aho-Corasick（AC）

- **何时启用**：已安装 `pyahocorasick`，且环境变量 **`EXPRESS_USE_AC` 未关闭**，且能从 Doris 拉取到非空字典并成功 `make_automaton` 时（详见 `express_ac.py` 中 `get_express_automaton`）。
- **字典内容**：对配置表（默认 `furniture_tms_busi__express_detail` 的 `expressNumber`）做 SQL 归一化后 **去重** 得到字符串集合；归一化表达式与校验路径一致：  
  `SUBSTRING_INDEX(TRIM(CAST(expressNumber AS CHAR)), ',', 1)`  
  即：**去空格、转成字符串、逗号取首段**（兼容库中「主单号,后缀」写在同一字段等形态）。
- **字典长度**：仅纳入 **字符长度** 在 `EXPRESS_AC_MIN_LEN`～`EXPRESS_AC_MAX_LEN`（默认 8～64）之间的串。
- **匹配办法**：在任务全文上运行 AC 自动机（子串多模式匹配）；对重叠命中做 **`_resolve_spans` 消解**（优先保留更长、互不重叠的区间，再按起点排序），得到最终 `(start, end, 单号串)`。
- **刷新**：默认每 **`EXPRESS_AC_REFRESH_SECS`（秒）** 可重拉字典；也可用 `EXPRESS_AC_FORCE_RELOAD` 强制单次重载。

#### 5.2 回退路径：候选抽取 + Doris `IN` 校验

- **何时启用**：AC 不可用（例如未安装 `pyahocorasick`、`EXPRESS_USE_AC=0`、字典为空或连接失败等）时，`model.py` 走此路径。
- **候选怎么来**（`express_doris.extract_express_candidates`）：在全文上收集若干类 **位置与字符串**，去重、长度不足 8 的丢弃，规则包括：
  1. **括号内文**：`【…】`、`（…）`、`(...)`、`[...]` 内，用 **字母数字与连字符** 组成的 token 正则 `[A-Za-z0-9](?:[A-Za-z0-9-]{7,39})` 找 **长度约 8～40** 的候选（token 总长度与括号内子串配合）。
  2. **物流词后的纯数字**：在「百世快运 / 顺丰 / 中通 / … / 运单号 / 面单 / 单号」等关键词后，取 **8～20 位连续数字**（可与中文、冒号、空格等分隔）；若该区间落在上述 **括号内文范围** 内则跳过，避免重复。
  3. **全文 token**：同一 token 正则在全文中扫描；与括号内文重叠的区间跳过。
  4. **独立纯数字段**：全文 **10～15 位** 数字段；若为 **11 位且符合大陆手机号** `1[3-9]…` 则 **丢弃**；与括号内文重叠则跳过。
- **校验办法**：将所有任务的全部候选单号 **去重分批**，用 Doris 查询  
  `WHERE SUBSTRING_INDEX(TRIM(CAST(expressNumber AS CHAR)), ',', 1) IN (...)`  
  落在库中的候选才输出为 **快递面单** 实体；未命中库的不产生该标签预测。

#### 5.3 小结（快递面单）

| 阶段 | 主路径（AC） | 回退路径 |
|------|----------------|----------|
| 候选/命中来源 | 字典中的单号必须是 **正文子串** 完全相等 | 先用 **规则从正文抽候选**，再用 **Doris 主单号归一化字段** 做存在性校验 |
| 典型优势 | 大批量字典下一次扫描快 | 不依赖 AC 构建成功时仍能「抽数字/ token + 库确认」 |
| 典型风险 | 字典未覆盖则不会命中 | 候选规则可能漏抽非常规写法；纯数字可能与业务单号撞车（已通过排除 11 位手机号等降低冲突） |

---

## 预测结果合并与排序

对每条任务文本，**手机号、发货计划单号、送装单号、合并单号、快递面单** 各规则独立产出若干 span，最后在 **`model.py` 中按 `start`、`end` 排序** 后写入 Label Studio 预测结构。不同标签在同一文本上 **可能重叠**；是否需要后处理去重由业务与标注规范决定，当前代码 **不做跨标签去重**。

---

## 测试

```bash
cd my_ml_backend
pip install -r requirements-test.txt
pytest test_api.py -q
```

（以你本地已安装 `label_studio_ml` 及测试依赖为前提。）

---

## 相关仓库文档

本文件只描述 **`my_ml_backend`**。整个仓库的通用说明见上级目录 `README.md`。
