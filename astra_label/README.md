# Astra 轨迹 progress 标注

输入一条参考轨迹及关键帧、一条待处理轨迹、共同的任务 instruction，使用 **Codex CLI 的 `gpt-6-astra`** 对待处理轨迹逐帧输出 `[0,1]` 的 progress。使用本项目已有 uv 环境，无新增依赖，不启动训练模型。

**所有目标 progress 都由 GPT-6 直接输出。** 关键节点约束、阶段内细分、失败回退、完成条件均在 `prompts.py` 中表达。代码不会按帧号插值、平滑、强制单调、吸附关键节点或将首尾重标定到 0/1。

## 代码与本地结果

仓库提交标注代码、测试、配置模板和可复用脚本。`outputs*/`、`runs/`、缓存、日志、`*.local.json` 和 `.env*` 不提交；这些本地文件不会因 Git 忽略而被删除。

- `run_drawer_batch.py`：六条 Close the drawer 轨迹的批处理与绘图入口，默认输出位置保持不变。
- `verify_drawer_api.py`：读取本机和 API 两组已生成的结果，校验原始分数并生成对比图；需要先在本地生成两组结果。运行 `uv run --no-sync python astra_label/verify_drawer_api.py`。

克隆仓库后，按 `api.example.json` 创建自己的 `api.local.json`，或设置 `ASTRA_LABEL_API_KEY`；不要把真实凭据写入配置模板。文中历史实验的输出文件仅存在于运行机器，不随代码仓库分发。

## 输入与运行

在项目根目录执行：

```bash
cp astra_label/example.json astra_label/job.json
# 编辑 job.json，替换 reference/target 的 category、id、关键帧及 instruction。

# 只读数据、检查参数、导出图像和参考 prompt，不调用模型。
UV_CACHE_DIR=/tmp/robometer-uv-cache uv run --no-sync python -m astra_label \
  --config astra_label/job.json --output astra_label/runs/my_job --prepare-only

# 正式标注；同一命令再次运行会复用已完成且请求完全相同的调用。
UV_CACHE_DIR=/tmp/robometer-uv-cache uv run --no-sync python -m astra_label \
  --config astra_label/job.json --output astra_label/runs/my_job
```

需要本机 `codex` 在 PATH 内。默认 `--auth-mode local` 使用已有 CLI 登录态；先用 `codex login status` 检查，尚未登录时运行 `codex login`。也可使用下述 `--auth-mode api`。遇到模型不可用直接报错，不自动替换模型。

`UV_CACHE_DIR` 只是让 uv 缓存写到可写位置，运行的仍是项目 `.venv`。`--no-sync` 避免重新解析或安装整套训练依赖。

## 选择本机登录或自定义 API

两个入口都支持 `--auth-mode local|api`：单轨迹的 `python -m astra_label` 和六条关抽屉轨迹的 `run_drawer_batch.py`。

- `local`（默认）：使用本机 Codex 已保存的登录态，保留其代理和用户配置。为保证显式模式选择生效，子进程会移除外部注入的 `CODEX_API_KEY` / `OPENAI_API_KEY` 等凭据覆盖；不修改父进程环境。
- `api`：使用自定义 Codex provider，`wire_api=responses`，通过专用环境变量传密钥。忽略全局用户配置并使用内存凭据存储，不调用 login/logout，不改写本机保存的登录凭据。API 模式默认清除子进程的 HTTP/HTTPS/ALL_PROXY 并设置 NO_PROXY，直接连接服务；如需继承代理，加 `--api-use-proxy`。

本机已配置的私密文件是 `astra_label/api.local.json`（权限 0600，已被 Git 忽略）。其结构见 `api.example.json`，默认地址是 `https://ckffai.com/v1`。真实 key 不写进 job 配置、prompt、命令参数、manifest 或缓存键；子进程输出写入日志前会将该 key 脱敏。更换 key 可编辑本地私密文件，也可使用环境变量 `ASTRA_LABEL_API_KEY` 覆盖。

可选参数：`--api-config /绝对路径/私密配置.json`、`--api-base-url https://服务地址/v1`、`--model 模型ID`、`--reasoning high`。私密配置文件只能包含 `base_url` 和 `api_key`。API 服务必须支持 Codex 的 Responses 流式协议、图像输入和结构化输出，仅兼容 Chat Completions 不足以保证可用。

在项目根目录启动六条关抽屉轨迹：

```bash
# 本机登录；原有结果/缓存继续有效。
UV_CACHE_DIR=/tmp/robometer-uv-cache uv run --no-sync python \
  astra_label/run_drawer_batch.py --auth-mode local

# 自定义 API；默认仍请求 gpt-6-astra，高推理等级。
UV_CACHE_DIR=/tmp/robometer-uv-cache uv run --no-sync python \
  astra_label/run_drawer_batch.py --auth-mode api
```

批量入口的 API 结果默认写入独立目录 `astra_label/outputs_api/close_the_drawer`，原来的 `astra_label/outputs` 保留本机实验；可用 `--output` 指定其他目录。单轨迹入口照常指定 `--config`、`--output`，API 结果使用 `--output astra_label/outputs_api/你的任务名` 并加 `--auth-mode api`。模式和 API 地址纳入缓存身份，不会将已有本机结果当成 API 结果。

**2026-09-28 最新复测：** `https://ckffai.com/v1` 和 `https://api.68886868.xyz/v1` 均可在关闭代理后访问；两者的 `/models` 均列出 `gpt-6-astra`、`gpt-6-luna`，`/chat/completions` 请求 Astra 均返回 HTTP 200。

现有 Codex 后端对 `ckffai.com` 的 **Astra / high / 图片输入 / JSON schema** 测试通过，保留它作为默认地址。`api.68886868.xyz` 的同一 Codex Responses 测试出现 `OutputTextDelta without active item`，并在 120 秒超时；其 Chat Completions 成功不代表本次 Codex 流式调用也成功。

原始请求结果及 Codex 日志位于 `astra_label/outputs_api/api_recheck_20260928T142624Z`。此前 Astra 无可用通道的 503 结果已被这次成功测试更新。服务返回的模型标识为 `gpt-6-astra`；这里验证的是所配置服务的调用结果，不验证服务商内部的模型部署。API 模式不向本机账号回退。

配置依据：[Codex 自定义模型提供商](https://learn.chatgpt.com/docs/config-file/config-advanced)、[配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)。

最小配置：

```json
{
  "reference": {
    "category": "你的参考子集目录名",
    "id": "参考记录的完整id",
    "keyframes": [12, 35, 60]
  },
  "target": {
    "category": "你的目标子集目录名",
    "id": "目标记录的完整id"
  },
  "instruction": "拿起杯子，放到盘子上并松开"
}
```

- `category` 是 vtools 中的子集目录名（例如 `abraranwar_libero_rfm_libero256_10`），不是 `robot/human` 等训练分组。
- `id` 精确匹配 Arrow 的 `id` 字段，**不是浏览器内部的 row 行号**。重复 ID 会报错。
- `keyframes` 是浏览器所显示的、处理后 NPZ 图像数组的 **0-based index**，严格递增且不能重复。不是原始视频帧号。
- 自动加入参考首帧 0 作为零分节点。以上例子得到 `0→0, 12→1/3, 35→2/3, 60→1`；关键帧必须体现有序的任务进展，最后一个必须代表完成。这里计算的只是**参考示例标签**，不会用于后处理目标分数。
- 若关键帧已含 0，不重复加入。仅 `[0]` 不能构成完成示范。
- `instruction` 省略时采用参考轨迹 `task`。参考和目标的原始 task 文本也传给 GPT-6，由模型判断是否同任务；无人工覆盖不同任务的开关。
- 数据根目录默认 `/mnt/workspace/dataset/robometer-1m`；可设置 `data_root` 或 `ROBOMETER_PROCESSED_DATASETS_PATH`。
- 可选 `reference.progress` 自定义参考分值，顺序对应**自动补 0 后**的全部关键帧，必须从 0 严格递增到 1。例如 `[0, 0.2, 0.7, 1]`。

`example.json` 是带占位符的模板，其关键帧仅为格式示例，不是对某条真实轨迹的人工标注。

## 标注流程

1. **定位图像**：沿用 `vtools/app.py` 的 Arrow stream/file 和历史路径重定位方式，从 `category/frames/<basename>.npz` 的 `frames` 数组读取；不向模型提供已有质量标签、语言向量或训练进度，避免泄漏。
2. **参考校准**：发送参考关键帧及各自前后邻帧、instruction、固定参考分值。GPT-6 为每个节点写出可观察的完成判据和阶段内判断方法，保存为 `rubric.json`。若参考不支持这些判据，立即停止。
3. **目标逐帧标注**：默认每批 16 个待标注帧；每次都先附参考图像与 rubric，再附目标图片，包括本批图、前后各 2 个上下文帧和 8 个全局概览帧。图片顶部显示 `REFERENCE/TARGET` 和原始 index，prompt 记录附件顺序。只给本批帧输出标签，上下文帧不重复标注。
4. **跨批上下文**：携带上一批末尾的标签；明确要求其仅作可纠错上下文，不能为平滑曲线而照抄。不会进行跨批后处理。
5. **验证与导出**：校验结构、帧覆盖、数值范围、关键帧引用是否合法。目标关键节点的语义和具体分数仍由模型判断，不用程序修正。每一个目标帧都有一次直接模型输出。

默认调用次数是 `1 + ceil(目标帧数 / batch_size)`，首个调用生成 rubric。程序开始时打印调用次数，便于估计规模。更多关键帧/邻帧/概览帧意味着每次调用图片更多；`prepare-only` 不消耗模型调用。模型调用使用 Codex 账户的可用配额/计费方式。

## 输出

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 输入定位、真实帧数、图像内容哈希、参考分值、调用次数 |
| `images/` | 附件图片，保留帧号并加参考/目标标识 |
| `reference_prompt.txt` | 参考校准的完整 prompt |
| `rubric.json` | GPT-6 对参考关键帧的阶段判据 |
| `calls/<阶段>/` | 实际 prompt、schema、附件哈希、CLI 命令、原始响应和日志 |
| `chunks/` | 每批模型返回的原始标注 JSON |
| `labels.json` | 全部逐帧标签、模型直接给出的关键帧匹配、参考 rubric |
| `labels.csv` | 相同逐帧标签，便于分析/训练转换 |
| `progress.png` | 原始 progress 曲线；橙点表示模型匹配关键帧，红叉表示待复核 |

每帧格式（仅结构示例，不是真实模型结果）：

```json
{
  "frame_index": 18,
  "progress": 0.3333333333333333,
  "matched_reference_keyframe": 12,
  "phase": "完成抓取",
  "confidence": 0.85,
  "evidence": "杯子已离开桌面且保持在夹爪中。",
  "needs_review": false
}
```

`matched_reference_keyframe` 表示模型判断当前状态对应哪个参考奖励节点，不匹配时为 `null`。它不是目标帧号。一个关键状态可能持续多帧，因而匹配可出现多次；未完成的节点可以完全没有匹配。`keyframe_matches` 只是这些模型字段的汇总，不做再次检测。置信度是模型自评，不是统计校准概率。

曲线连线只用于展示，没有产生额外标签。**没有强制目标首帧为 0、末帧为 1，也没有保证单调。** 遮挡、阶段退化、任务未完成等都在 prompt 中要求模型自行判断并解释。

## 参数和重试

可选配置：`batch_size=16`、`context_frames=2`、`overview_frames=8`、`reference_neighbors=1`、`max_image_side=768`、`model="gpt-6-astra"`、`reasoning="high"`、`timeout=900`、`codex_bin="codex"`。图像只按最大边等比例缩小，不放大；默认 256px 源图保持原图分辨率。将 `overview_frames` 或 `reference_neighbors` 设为 0 可减少上下文图片。

失败时不会自动产生替代标签。查看 `calls/<阶段>/stderr.log`、`events.jsonl` 和 `response.json`。修复登录、网络等问题后，用相同命令继续。调用超时是**每次调用**的超时。

若出现 `failed to initialize in-process app-server client: Read-only file system`，是外层运行环境不允许 Codex 写入自己的本地状态；需在允许 Codex 正常初始化的终端运行。脚本的 `--sandbox read-only` 限制的是模型执行的命令，不会免除 CLI 本身对状态目录的写入需求。

缓存键包含模型、推理设置、完整 prompt、schema 和图像内容；缓存返回后仍进行标签校验。若想重试一次模型拒绝/格式有误的已有响应，可删除对应 `calls/<阶段>/validated_transport.json` 后重跑；原始响应和日志在下一次该调用时更新。改配置或数据时使用新输出目录，避免混入旧结果。不要并发写同一输出目录。

第一版旨在研究模型标注质量：逐帧看图和分批上下文无法保证语义完全一致，尤其是长时序、遮挡和多子任务场景。建议人工复核 `rubric.json`、关键节点和 `needs_review` 帧，再将标签接入训练。此流程不改写原始数据集。

## 离线验证

```bash
UV_CACHE_DIR=/tmp/robometer-uv-cache MPLCONFIGDIR=/tmp/robometer-matplotlib \
  uv run --no-sync python -m unittest discover -s astra_label/tests -v
```

测试使用临时 Arrow/NPZ 和模拟远程模型响应，验证 ID 定位、关键帧输入、逐帧覆盖、错误响应、回退分值保留和最终文件导出；不消耗模型调用。真实数据可以单独用 `--prepare-only` 验证读取。

开发验证：9 项离线测试通过；已用本地两条 32 帧 LIBERO 轨迹验证读取及 `prepare-only`；已通过真实 `gpt-6-astra` 调用验证图片附件和结构化 JSON（人工红色色块返回 `{"color":"red"}`）。这不等同于验证真实机器人轨迹的标注准确率，仍需用你选择的关键帧进行正式标注和复核。

调用依据：[Codex 非交互运行与结构化输出](https://learn.chatgpt.com/docs/non-interactive-mode)、[GPT-6 Astra 模型说明](https://developers.openai.com/api/docs/models/gpt-6-astra)。本机 CLI 的 `codex exec --help` 可确认 `--image`、`--model`、`--output-schema` 等参数。
