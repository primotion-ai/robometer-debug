# B200：27B / 397B Robometer 实验

三个入口均支持 `27b` 和 `397b`，模型和机器规模没有绑定关系：

| 入口 | 用途 | W&B | 默认 GPU 数 |
| --- | --- | --- | --- |
| `train_single_b200.sh` | 单机正常训练，完整数据及 OOD 验证 | online | 8 |
| `train_multi_b200.sh` | 多机正常训练，完整数据及 OOD 验证；用 NNODES 指定节点数 | online | 默认 5 × 8 |
| `debug_b200.sh` | 单机短程排错，默认 3 个 optimizer step | disabled | 8 |

所有脚本使用 uv 创建的仓库 `.venv`，不会在启动时自动升级依赖。多机脚本自动切换到仓库目录，设置 `VIRTUAL_ENV` 和 `PATH`，直接执行 `.venv/bin/python`，无需在作业界面先激活环境，也不要求容器 PATH 中能找到 uv。可用 `ROBOMETER_VENV` 指定其他预先准备好的环境目录。单机和调试入口仍使用 `uv run --no-sync`。新环境在提交前准备一次：

```bash
uv sync --locked --extra b200 --extra b200-kernels
```

`b200` 固定 Transformers 5.5.0，支持 Qwen3.5 dense/MoE，也满足现有 Unsloth 的版本上限。新实验关闭 Unsloth。保留 `uv.lock`，让各容器环境一致。

挂载已有 `.venv` 时，容器还必须具备兼容的 Python 3.10 和系统库；本机 `.venv/bin/python` 链接到 `/usr/bin/python3`，仅挂载目录不会把基础解释器一起打包。不要让所有节点同时对同一个共享 `.venv` 执行 `uv sync`。平台示例中的 `CUDA_VERSION=12.3.1` 不能用来判断实际训练环境；本项目使用 Torch 2.8/cu128，启动检查会打印实际 Torch 版本、Torch CUDA runtime 和可见 GPU 数。保留平台注入的 NVIDIA/CUDA、`LD_LIBRARY_PATH` 和 NCCL/RDMA 环境变量，不用本机值覆盖。

## 数据、模型和凭据

- `27b` 配置：`robometer/configs/qwen27b.yaml`，权重 `/mnt/base_model/Qwen3.8-27B`。本地模型实际是 Qwen3.8-27B，复用 `qwen3_5` 架构，不能用 Qwen3-VL 类加载。
- `397b` 配置：`robometer/configs/qwen397b.yaml`，权重 `/mnt/base_model/Qwen3.5-397B-A17B`，架构 `qwen3_5_moe`。
- `ROBOMETER_PROCESSED_DATASETS_PATH` 默认 `/mnt/dataset/robometer-1m`。训练选择 `rbm-1m-id`，验证选择 `rbm-1m-ood`；`/mnt/dataset/rbm-1m-ood` 是指向前者子集的链接集合，不必改数据根目录。
- 正常训练读取 `.secrets/wandb_api_key`（权限 600，目录权限 700，Git 和 Docker 构建上下文均忽略）。也可通过 `WANDB_API_KEY_FILE` 指向容器挂载的 secret 文件，或使用平台注入的 `WANDB_API_KEY`。不要将密钥写入作业命令、YAML、镜像或提交到 Git。
- 调试不读取 key 文件，并从子进程环境移除 `WANDB_API_KEY`。W&B、代码上传、Hub 上传及模型保存均关闭。
- `WANDB_PROJECT` 默认 `robometer-qwen-b200`；`WANDB_ENTITY` 留空表示使用 key 所属账号的默认 entity，可通过环境变量指定团队。
- 全作业仅全局 rank 0 初始化 W&B、上传指标和写入 wandb_info.json；五机只创建一个 run。主进程判断使用已初始化进程组的全局 rank，初始化前使用 RANK，不使用每节点都会重复的 LOCAL_RANK。训练指标保留原来的跨卡汇总。已运行的旧进程不会热更新此修复，需在下次启动或从 checkpoint 恢复时生效。
- 启动前会检查完整权重分片及文件长度、processor、数据缓存和可见 GPU。397B 下载未完成会明确报错退出。

## 单机正常训练

```bash
RUN_NAME=rbm_qwen27b_single_001 bash run_scripts/train_single_b200.sh 27b
RUN_NAME=rbm_qwen397b_single_001 bash run_scripts/train_single_b200.sh 397b
```

默认完整语言主干 + 三个奖励头训练，冻结视觉编码器，SDPA、gradient checkpointing、FSDP1 FULL_SHARD。27B 保持 BF16 混合精度、每卡训练 batch=16；397B 使用下述独立纯 FP16 配方、完整逐层 checkpoint、每卡训练 batch=4。两者均为梯度累积=1、8 帧/轨迹；27B 默认训练 5,000 step、学习率 `1e-5`，397B 默认训练 3,000 step、学习率 `5e-6`。一个 batch 样本是一个 preference pair，包含两条轨迹。评估 batch 仍为每卡 1，两个模型都每 500 step 评估一次。

两个 B200 配方都设置 `data.dataset_num_repeats=1`，按真实数据集长度洗牌，Trainer 自动重复 epoch 直到各配方的 `training.max_steps`：27B 为 5,000，397B 为 2,000。原项目默认用 `RepeatedDataset(..., num_repeats=1000)`：全量过滤后的 1,041,377 条轨迹被虚拟扩大成 1,041,377,000 条，随机 sampler 会为每个 rank 建立十亿项 Python 索引列表，导致巨额 CPU 内存和 GC 停顿。纯 CPU 复现中该列表占约 38.9 GiB，GC 耗时 9.1–15.7 秒，而百万项列表约 0.008 秒。该 GC 修复保留自动 GC，不改变 batch、帧数、学习率或当时的 max_steps；采样顺序和 epoch 定义改变为“每个真实 epoch 洗牌一次”。其他旧配方的默认重复倍数仍为 1000，只有 B200 配方覆盖为 1。

旧配置的 checkpoint 恢复到新采样方式时，模型/优化器步数仍可恢复，但数据跳过位置不与原先十亿项排列等价；本次调试没有保存 checkpoint，正式实验应从新配置开始。CPU 复现记录位于 `logs/rank_wait_full_native/list_gc_reproduction.json`。

不加载模型/GPU即可快速验证 GC 随索引列表长度增长：`uv run --no-sync python scripts/benchmark_sampler_gc.py`（默认测百万/千万项，约 1 秒）。显式加 `--repeats 1000` 可复现原始十亿项列表，但会占约 39 GiB 主存。脚本使用顺序整数，省略 randperm 计算，只隔离列表的 GC 扫描成本。

修复后全量验证（2026-09-29）：8 卡、27B、每卡 batch 16、8 帧、尺寸 null、worker 8、pin memory 开启，正常取不同 batch，保留自动 GC；无原生/CUDA profiler，无 W&B、保存或评估。3 个 optimizer step 正常退出（code 0），无 OOM。所有 rank 第 2 步前后向为 17.47–17.57 秒，第 3 步为 17.00–17.47 秒，前向约 5.2 秒；上述前后向计时不含 optimizer.step。rank 0 首批取数 8.16 秒，后续 0.027/0.010 秒；所有 rank 首批为 8.16–15.46 秒。GC 最长 0.728 秒，没有再出现十几秒长停顿。训练部分共 75.72 秒（含首批与预热），数据初始化仍约 352 秒。

此次 PyTorch peak allocated 为 78.21–79.10 GiB/卡，reserved 为 95.03–95.95 GiB/卡；nvidia-smi 采样峰值约 96.95–97.87 GiB/卡。训练结束前各主进程 RSS 约 18.6–20.1 GiB，此前大索引列表场景约 57–59 GiB。输入、GPU 数与旧七卡三步测试不完全相同，不将前后比值作为严格吞吐加速比；十亿项列表的独立 GC 复现、同批次 GC 开关对照以及此修复验证共同支持根因判断。完整记录在 `logs/qwen27b_bs16_frames8_native_8gpu_w8_repeat1_20260929_050737/`，同名加 `_monitor` 的目录有退出码和 GPU 采样。

正式训练的 `data.resized_height`、`data.resized_width` 均为 `null`，与原始 4B 配方一致：读取缓存图片，再由视觉处理函数和模型 processor 按尺寸约束处理，不额外强制拉成 224×224。不指定尺寸不代表完全不缩放，不同 backbone 的 processor 也不保证产生相同视觉 token 数。27B 已完成 7 卡、每卡 batch=16、8 帧、不指定尺寸的短程验证，结果见文末；397B 尚未完成全模型显存验证。

397B 单机按约定保留全参训练，默认 `FSDP_OFFLOAD_PARAMS=true`，不自动切换 LoRA。此前 BF16 混合精度配置保留 FP32 主权重/梯度/AdamW 动量，按模型名称粗估约 6.35 TB；当前改用下面的纯 FP16 配方，理论持久状态约减半，但仍不能承诺单机跑通。主机物理内存不等于容器内存配额，加载峰值和优化器更新峰值仍需单独验证。CPU offload 模式限制梯度累积为 1。多机默认不 offload，可显式设置 `FSDP_OFFLOAD_PARAMS=true/false`。

参数可以放在命令末尾覆盖，例如：

```bash
RUN_NAME=rbm_qwen27b_single_bs2 bash run_scripts/train_single_b200.sh 27b \
  training.per_device_train_batch_size=2 training.gradient_accumulation_steps=4 \
  data.max_frames=8 training.learning_rate=1e-5
```

有效 batch = 节点数 × 每节点 GPU 数 × 每卡 batch × 累积步数。正式训练默认累积步数=1：27B 每卡 batch=16，单机 8 卡、五机 40 卡、八机 64 卡分别为 128、640、1024 对轨迹/更新；397B 每卡 batch=4，分别为 32、160、256 对。27B 的 5,000 步在单机和五机分别累计处理 64 万和 320 万对采样样本；397B 的 3,000 步在单机、五机和八机分别为 9.6 万、48 万和 76.8 万对。上述计数包含重复采样及增强，不能等同于独立原始轨迹数；学习率没有随 batch 自动放大。

FSDP1 梯度累积期间的 `no_sync` 会保留未分片梯度，显著增加峰值显存；397B 建议保留累积步数 1，先调每卡 batch。

### 397B 独立纯 FP16 配方

`qwen397b.yaml` 单独设置 `model.torch_dtype=float16`、`training.pure_fp16=true`；启动器为 397B 选择 `distributed/b200_397b_fp16_fsdp.yaml`。参数、梯度分片、AdamW 的 exp_avg/exp_avg_sq 使用 FP16，不保留 FP32 主权重。`training.fp16=false` / `bf16=false` 和 Accelerate `mixed_precision: 'no'` 是有意关闭 AMP，防止 FSDP 自动将主权重升成 FP32；它们不表示模型使用 FP32。4B/27B 不启用 pure_fp16，保持既有配置和优化器行为，共享 b200_base.yaml / b200_fsdp.yaml 没有修改。

梯度检查点通过 `gradient_checkpointing=true`、`gradient_checkpointing_backend=fsdp` 开启。FSDP 对每个 Qwen3_5MoeDecoderLayer 和 Qwen3_5MoeVisionBlock 使用非重入 checkpoint wrapper，语言层内部的注意力和 MoE 都随层重计算。Trainer 的同名 checkpoint 开关在此路径关闭，避免重复包装；视觉编码器仍冻结。启动时检查所有目标 block 的 wrapper 覆盖，第一次更新后各 rank 写 `pure_fp16_rank_<rank>.json`，验证参数、梯度和两个 AdamW 动量 dtype。

397B 专用 AdamW 使用 epsilon=1e-4（1e-8 在 FP16 中会舍入到 0），foreach=false、fused=false，避免 foreach 的额外整组参数临时张量。未启用 AMP GradScaler。优化器 step 计数器仍是标量 FP32；模型内部的归一化、路由 softmax、FLA 累加和 loss 可以保留 FP32 运算，这不是额外的 FP32 主权重或参数大小的优化器状态。

2026-10-03 首步 NaN 排查：对真实预训练权重做 64-token 短文本逐层前向（无 optimizer/反向/FSDP），在语言层索引 22 的 shared expert `down_proj` 复现 FP16 Inf。相同输入下，gate/up 投影及中间乘法均有限，但 FP32 输出投影最大绝对值约 280,501，超过 FP16 的 65,504；乘共享专家门控后最大仅约 38.6。现仅针对 FP16 Qwen3.5 MoE，将共享专家中间乘法、输出投影以及门控合并留在 FP32，合并路由专家输出后转回 FP16。所有参数、参数梯度和 AdamW 状态仍为 FP16，不增加持久 FP32 主权重；BF16 的 4B/27B 路径不变。新增的是局部临时 FP32 张量，不能保证完整模型的其他层、长输入和反向都不会溢出。

纯 FP16 比混合精度更易发生溢出、梯度/二阶动量下溢以及微小参数更新舍入；调整 epsilon 不能消除这些问题。这里是用户选择的省显存实验配方，不等价于已经验证完整 397B 收敛。按本地语言主干约 4019 亿参数、每参数 8 字节持久状态计算，约 3.22 TB，40 卡约 74.9 GiB/卡，未计激活、冻结视觉、通信聚合和临时缓冲，不能当作实际峰值承诺。

现有 `train_single_b200.sh 397b`、`train_multi_b200.sh 397b` 和 `debug_b200.sh 397b` 自动采用该配方。397B 正式训练默认每卡 batch=4，只在 qwen397b.yaml 覆盖；调试入口仍默认 batch=1，可通过命令参数覆盖。397B 正式训练 3,000 步，每 1,000 步保存一次，保留全部 3 份 checkpoint；帧数、FLA/SDPA 和每 500 步评估不变。已有混合精度实验请勿直接复用同名输出目录；本改动不保证旧精度 checkpoint 切换后的数值等价。

2026-10-03 的 40 卡 batch=8 数值诊断已通过前向并进入反向，但在 FLA/Triton 反向内核自动调优时 OOM；日志确认 87 个 FSDP checkpoint wrapper 生效。现将 397B 正式默认 batch 降为每卡 4、全局 160，累积步数仍为 1。batch=4 尚未完成完整模型容量验证，应至少连续跑 3 个 optimizer step，覆盖 AdamW 状态分配后的反向峰值；不能只用首步前向判断显存是否足够。

若首次前向后出现 `average_precision_score: Input contains NaN`，表示成功率预测已经含 NaN，指标函数只是发现问题。`ConstantInputWarning` 则表示 Spearman 输入为常量，本项目将该相关系数记为 0，不会把它写回模型预测。纯 FP16 训练现会在奖励头输出处直接检查 NaN/Inf 并报错，不将异常预测替换为 0。用以下命令在平台的全部 5 个节点启动一次数值诊断：

```bash
ROBOMETER_CHECK_NUMERICS=1 RUN_NAME=rbm_qwen397b_bs4_diag_001 \
  bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 397b \
  training.max_steps=3
```

该开关仅支持 397B：保留正式 batch=4、8 帧、原图尺寸、40 卡和精度配置，默认最多 1 个 optimizer step（上例显式改为 3 步），强制关闭 W&B、评估和 checkpoint 保存。每个叶子模块及目标 block 检查输出，发现 NaN/Inf 立即报出模块路径并写 `logs/<RUN_NAME>/first_nonfinite_rank_<rank>.json`；这会增加 GPU 同步开销，仅用于定位异常。它不修复数值问题，也不能保证不同样本的错误必然复现。移除该环境变量即恢复正式训练。FSDP 日志同时报告 checkpoint wrapper 数量；Transformers 的 `gradient_checkpointing=False` 不代表 FSDP checkpoint 未开启。

如果作业长期停在首步、平台无法进入终端，可重新提交阶段诊断（先停止旧作业，使用新 RUN_NAME）：

```bash
ROBOMETER_TRACE_PROGRESS=1 ROBOMETER_CHECK_NUMERICS=0 \
RUN_NAME=rbm_qwen397b_bs4_trace_001 \
  bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 397b \
  training.max_steps=3
```

阶段诊断同样禁用 W&B、评估和保存，保留每卡 batch=4 和实际输入，但关闭逐模块 finite 检查以减少同步开销。每个 rank 每 30 秒打印 `[ROBOMETER_PROGRESS]` 并追加 `progress_rank_<rank>.jsonl`，记录步数、前向/反向/优化器阶段、最近模块与 PyTorch 显存；每 120 秒写 `stacks_rank_<rank>.log`。模块事件记录的是 CPU 执行位置，不是 CUDA 同步完成时刻；应结合所有 rank 的栈判断计算、编译或通信等待。首次更新完成后才会出现 `pure_fp16_rank_<rank>.json`，不能仅凭 GPU 利用率或 DATA SOURCE 日志判断训练已完成一步。移除诊断开关和 max_steps 覆盖后才恢复正式 2,000 步训练。

验证记录：4 项纯 FP16 配置/优化器测试、原有梯度累积回归通过；27B/397B 五机命令预演分别选择原有/独立 FSDP 配置。两张 B200 上的小型 Qwen3.5 MoE（FLA + SDPA、冻结视觉、三个奖励头、2 个 decoder + 1 个 vision block）完成 2 步真实更新和分片保存，再开启 CPU offload 从 checkpoint-1 恢复到 step 3，均正常退出。测试检查各 rank 的参数、梯度、AdamW 两个动量为 FP16、每一步数值有限且确实发生参数更新、所有目标 block 被 checkpoint 包装；不上传 W&B。记录在 `logs/validation_397b_pure_fp16/`。这不代替完整 397B 权重加载、显存容量或训练收敛验证。

可在两张空闲 GPU 上重跑小模型集成测试（不加载 397B）：

```bash
CUDA_VISIBLE_DEVICES=0,1 WANDB_MODE=disabled HF_HUB_OFFLINE=1 \
  PURE_FP16_TEST_DIR=/tmp/rbm-pure-fp16-check \
  .venv/bin/python -m accelerate.commands.launch \
  --config_file robometer/configs/distributed/b200_397b_fp16_fsdp.yaml \
  --num_processes 2 --num_machines 1 --main_process_port 29736 \
  --num_cpu_threads_per_process 2 tests/run_pure_fp16_fsdp.py
```

### 397B 异常 batch 跳过（2026-10-04）

397B 默认 `training.skip_anomalous_updates=true`、`training.skip_loss_threshold=10.0`。任一 rank 的本地 batch loss 大于 10、loss/头输出出现 NaN/Inf，或全局梯度范数非有限时，所有 rank 一起丢弃这批梯度，不执行 AdamW（包括 weight decay 和状态更新），不推进 scheduler/global_step。有限梯度仍按原来的 `max_grad_norm=10` 裁剪；梯度范数超过 10 本身不是跳过条件。FSDP 仍完成当前反向以清理分片状态，因此跳过更新不意味着省掉该 batch 的全部计算。

总步数 2000、warmup 200、保存 1000/2000 全部按**成功更新**计数。跳过后自动取下一批，必要时进入下一轮数据。rank 0 在 `logs/<RUN_NAME>/skipped_updates.jsonl` 单独追加异常记录，包括尝试次数、成功更新数、累计跳过数、学习率、每个 rank 的 loss/梯度范数/异常输出位置和数据来源。checkpoint 中的 `skipped_update_state.json` 保存实际消费的 epoch/batch 游标，恢复时不会把跳过的 batch 当成未消费数据。

这一路径目前仅支持 397B pure FP16、`gradient_accumulation_steps=1`；4B/27B 保持原来的训练循环。逐模块 `ROBOMETER_CHECK_NUMERICS=1` 诊断仍会直接抛错，正式训练使用 0。异常跳过可避免坏梯度写入权重，不会修复已经损坏的参数，也不处理 OOM 或通信失败。

本机两卡 FSDP + checkpoint + FLA 小型 MoE 注入验证：只在 rank 1 注入大 loss、NaN loss、异常梯度、NaN 头输出，共 6 次尝试、4 次全局跳过、2 次成功更新。逐项断言跳过时参数及 AdamW 状态不变、scheduler 仅推进 2 次；跨 epoch 及 checkpoint 恢复验证通过。完整 40 卡 397B 仍需新作业验证。

```bash
ROBOMETER_TRACE_PROGRESS=0 ROBOMETER_CHECK_NUMERICS=0 \
RUN_NAME=rbm_qwen397b_5nodes_2000steps_002 \
bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 397b
```

### 线性注意力内核

两个模型共用 `model.linear_attention_backend=fla`，要求 `fla-core[cuda]==0.5.2` 和 `causal-conv1d==1.7.0`。缺少可用内核会报错，避免静默回退。27B 有 48 个 GatedDeltaNet 层，397B 有 45 个；两者使用同一实现方案，但需要分别验证全模型训练。`attn_implementation=sdpa` 只控制普通/视觉注意力，不代表线性注意力也使用 SDPA。

397B 另设 `model.experts_implementation=eager`：当前 Torch 2.8 的 `torch._grouped_mm` 在 B200/SM100 报错，只支持 SM90。此设置是兼容性回退，不是 MoE 性能优化；更快的专家计算应在兼容的后端/环境中另测。27B 没有这项 MoE 限制。

2026-10-03 的五机阶段诊断显示首步反向持续推进，但从层 59 到层 29 耗时约 56 分钟，每层约 125 秒；`shared_expert_gate` 只是最近完成的重计算模块，Python 栈停在 C++ autograd 引擎，不能将耗时全部归因于该门控或 NCCL。真实尺寸（512 专家、hidden=4096、intermediate=1024、8192 token、top-k=10）两卡 FSDP 对照发现旧 eager 的逐专家权重索引会触发 1024 次 SelectBackward，反复分配/清零整个打包专家参数大小的梯度。现仅在 FP16 Qwen3.5 MoE + eager 加载路径中改为一次 `unbind` 权重视图，反向统一拼接梯度；权重键、形状、dtype、路由和训练目标不变。相同两卡 profiler 基准反向约 13.0 秒降至 1.1 秒；八卡同样配置下旧实现约 33.35–33.47 秒，优化后约 1.85–2.04 秒。两者均为单个真实尺寸专家层（含 FSDP、无完整 backbone），峰值 allocated 未明显变化，尚不能作为完整五机训练的加速比。FP64/FP16 的输出、输入/路由/参数梯度以及未激活专家零梯度对照通过；两卡小型 MoE 的 FSDP、完整 checkpoint、纯 FP16 AdamW 两步更新及分片保存通过。正在运行的旧进程不会热更新此优化。

对照原生实现时，在任一启动脚本末尾追加 `model.linear_attention_backend=torch`。`auto` 则保留 Transformers 自动选择行为。开关作用于模型实例，不修改 `.venv` 内的 Transformers 源码。单机、八机和调试入口均继承共同配置；batch、帧数、学习率和训练目标不因内核选择而改变。

没有启用 FlashAttention 4：与现有 protobuf 约束兼容的试验版本 `4.0.0b10` 在 B200 上不支持语言模块的 256 维 head，视觉模块的 72 维 head 反向编译也失败。因此普通/视觉注意力暂保留 SDPA，线性注意力优化不应被解读为解决了所有同步等待。

SDPA 本身会自动选择 PyTorch 内置的优化 kernel，本机 trace 中视觉路径已经使用 Flash kernel；它仍有逐段 Python 调用及 `lengths.tolist()` 同步。外部 FlashAttention 的 varlen 路径与这个行为不同。参考 [PyTorch 2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html) 和 [Transformers 5.5 注意力后端](https://huggingface.co/docs/transformers/v5.5.0/en/attention_interface)。

可重跑 CUDA 数值检查（指定一张空闲卡）：

```bash
CUDA_VISIBLE_DEVICES=0 RBM_TEST_CUDA_KERNELS=1 uv run --no-sync python \
  -m unittest discover -s tests -p test_qwen_cuda_kernels.py -v
```

该检查包含两种架构的小模型多模态前后向，以及 48/64 value heads、128 head dimension 的线性注意力输出和梯度对照，不代替完整模型/八机训练验证。

### CPU 数据读取并发

正式训练的 `data.dataloader_num_workers=8` 是每个 GPU/rank 的 worker 数：单机 8 卡共 64 个，八机共 512 个。保持 `data.dataloader_persistent_workers=true` 和 pin memory；PyTorch 在有 worker 时默认每个 worker 预取 2 个 batch。`OMP_NUM_THREADS` 默认 8，控制每个训练进程的 CPU 计算线程，与 DataLoader worker 数是两个独立参数。

当前主机有 112 个物理核、224 个逻辑 CPU；每卡 8 个读取 worker 是提高并发后的起点，尚未证明吞吐最优。容器 CPU 配额、磁盘吞吐及 `/dev/shm` 容量也会限制效果，可通过 `data.dataloader_num_workers=4/8/16` 分别实测。不要简单按集群总 CPU 数给每个 rank 配 worker。

调试默认仍为 0 个 worker，方便直接看到数据异常。测试正式读取并发时追加 `data.dataloader_num_workers=8 data.dataloader_persistent_workers=true`。启动阶段的数据过滤另用硬编码 `num_proc=8`，每个 rank 独立执行；提高 DataLoader worker 数不改变这部分初始化耗时。

## 平台容器任务：多机正常训练

在平台建立 **NNODES 个容器实例，每实例 8 张 B200，每实例执行一次启动命令**。脚本在每个容器内启动 8 个 GPU worker；5 台为 40 个训练进程，8 台为 64 个。不要让平台先为每张卡执行一次本脚本，以免重复拉起进程。

如果通过云平台作业界面提交，选择能在每个节点执行一次用户命令的 PyTorch/自定义分布式任务。27B 选择总共 5 个训练节点、每节点 8 张 B200。如果界面区分 Master/Worker，且 Master 也执行训练命令并分配 8 卡，则应是 1 Master + 4 Workers。平台应该注入 `NNODES=5`、`NPROC_PER_NODE=8`、`WORLD_SIZE=40`、各节点不同的 `NODE_RANK=0..4`，以及一致的 `MASTER_ADDR` 和 `MASTER_PORT`；表格中 1 的示例值不是五机作业的配置。

作业界面的“启动命令”直接填写以下内容，脚本会自行加载 uv 环境、切换工作目录并执行训练。挂载路径与本机一致时无需修改模型、数据或环境路径。输出默认位于 `/mnt/liuzihao/project/robometer/logs/<RUN_NAME>`；此目录必须是所有节点共同可写的共享存储，同名的各节点本地目录不满足分片保存要求。

```bash
# 作业 A：27B，界面分配 5 节点 × 8 卡，平台在所有节点执行同一命令。
RUN_NAME=rbm_qwen27b_5nodes_003 bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 27b
```

```bash
# 作业 B：397B，界面分配 8 节点 × 8 卡，平台注入 NNODES=8 / WORLD_SIZE=64。
RUN_NAME=rbm_qwen397b_8nodes_001 bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 397b
```

平台需要注入的变量和挂载：

| 项目 | 值/含义 |
| --- | --- |
| `MASTER_ADDR` | rank-0 容器在所有节点可达的 IP 或 DNS |
| `MASTER_PORT` | 默认 29500，节点间可达，同一作业保持一致 |
| `NODE_RANK` / `GROUP_RANK` / `RANK` | 按此优先级读取，必须是 **节点编号 0…NNODES-1**；5 台为 0…4 |
| `NNODES` | 读取平台值；未提供时默认 5，表示训练节点数，不从 `WORLD_SIZE` 推断 |
| `NPROC_PER_NODE` | 支持整数、auto、gpu；auto/gpu 按容器实际可见 GPU 数启动，8 张可见卡对应 8 个进程，不回退 CPU 训练 |
| `GPUS_PER_NODE` | 兼容旧命令；NPROC_PER_NODE 缺省或为 auto/gpu 时可显式指定 GPU 进程数；两者均为整数时必须一致；两者都没有时默认 8 |
| `WORLD_SIZE` | 按平台提供的含义，若存在必须等于 NNODES × 每节点 GPU 数，不一致立即报错 |
| `RUN_NAME` | 所有容器必须相同，不同实验使用不同名称 |
| `WANDB_API_KEY_FILE` | secret 文件挂载路径，或直接由平台注入 `WANDB_API_KEY` |
| 数据/权重 | 所有节点相同路径可读，权重必须下载完整 |
| `training.output_dir` | 默认 `./logs`，**多机 checkpoint 必须使用各节点可写的共享存储** |

如果平台的 `RANK` 是 GPU 全局 rank 而非节点 rank，应调整平台为“一容器一节点”的入口模式，并显式传 `NODE_RANK`。脚本检测到 `LOCAL_RANK` 时会拒绝嵌套启动。

不要在界面命令外再套 `torchrun`、`accelerate launch` 或 `mpirun -np "$NP"`；该脚本已经通过 Accelerate/torchrun 在每个节点拉起 GPU worker。MPI 的 hostfile、`kubexec.sh` 和 `NP` 不参与本入口的拓扑计算，单纯存在这些变量不会触发 MPI。若平台只在一个 MPI launcher 容器执行用户命令、其余 worker 不执行，则需要另一种 MPI 分发入口，不能直接套用此命令。此区别取决于平台的任务类型，不能仅凭后端使用 K8S 判断。参考 [Kubeflow PyTorch 环境说明](https://trainer.kubeflow.org/en/latest/legacy-v1/getting-started.html) 和 [MPI launcher/worker 机制](https://blog.kubeflow.org/integrations/operators/2020/03/16/mpi-operator.html)。

先用同一作业配置做不训练、不上传的环境检查：

```bash
RUN_NAME=rbm_qwen27b_5nodes_check bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 27b --check-only
```

`--check-only` 不启动分布式通信，不能验证跨机网络。`--dry-run` 只生成命令，不做模型/数据完整检查或读取密钥；NPROC_PER_NODE=auto/gpu 且未显式指定 GPUS_PER_NODE 时，仍需查询本容器可见 GPU 数。五台正式启动的每个节点应打印 `5 nodes x 8 GPUs`、自身 node rank、`total GPU workers: 40`；27B 应打印 `Effective batch: 640`，397B 为 `Effective batch: 160`。若 WORLD_SIZE 或 NPROC_PER_NODE 仍是占位值 1，先修正平台作业参数；脚本不会静默忽略或覆盖这些冲突。正常训练仍读取忽略上传的 `.secrets/wandb_api_key` 或平台挂载的 secret，并开启 W&B；单机调试入口仍不上传。

脚本进入后先打印 `[robometer launcher] entered` 和必要的分布式变量，不输出完整环境或密钥。若 worker 日志只有 `nslookup ... waiting for master`，先检查 master 的状态与主容器日志；这段等待不在本脚本中，可能来自平台初始化容器。曾观察到 master 因平台注入 NPROC_PER_NODE=auto 被旧版整数检查拒绝而反复退出；当前版本已支持 auto，无需将其手动改为 8。仅凭 worker DNS 等待不能区分 master 启动失败与平台 DNS 故障。

3.2T 链路不需要写入 FSDP 配置。在全部训练 rank 上 FULL_SHARD（5 台为 40 个，8 台为 64 个），参数、梯度和优化器状态均全切分。保留平台配置的 NCCL / RDMA 环境；不硬编码网卡名，也不禁用 IB。如平台要求，可注入 `NCCL_SOCKET_IFNAME`、`NCCL_IB_HCA` 和 `NCCL_IB_GID_INDEX`。需要容器能访问平台分配的 RDMA 设备，首次可加 `NCCL_DEBUG=INFO` 确认实际使用的链路；这里尚未验证跨机 NCCL 带宽。

5 台 27B 的正式配置是每卡 batch 16、累积 1、全局 batch 640，学习率仍为 1e-5，训练 5,000 步，每 1,000 步保存一次，保留全部 5 份 checkpoint，每 500 步评估。启动器在未注入 NNODES 时默认 5 台；平台值优先，作业界面也必须选择 5 个训练节点。新实验使用上例的 `rbm_qwen27b_5nodes_003`，不复用已有的 001/002 输出目录。已对节点编号 0 和 4 做过 --dry-run，生成 40 进程的正确命令；这不代表已经验证跨机通信、正式 OOD 评估或大模型 checkpoint 保存/恢复。单机八卡短测约 98 GiB/卡，支持五台全切分容量可行的判断，但多机每步时长和共享存储吞吐仍需实测。固定每卡 batch 时，每卡计算量没有缩小，不能将单机 step 时间直接除以 5。

## 单机调试

```bash
# 本次机器的 GPU 7 有其他任务；示例使用空闲的 0、1、2、3。
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 DEBUG_STEPS=3 \
  bash run_scripts/debug_b200.sh 27b

# 权重齐全后可尝试 397B 全参调试；本机容量限制见上文。
DEBUG_STEPS=3 bash run_scripts/debug_b200.sh 397b

# 逐渐增大帧数、分辨率和 batch，测显存。
CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS_PER_NODE=4 DEBUG_STEPS=5 \
  bash run_scripts/debug_b200.sh 27b \
  data.max_frames=8 data.resized_height=224 data.resized_width=224 \
  training.per_device_train_batch_size=1
```

默认仅使用本地 MetaWorld 训练子集、每卡 batch=1、2 帧/轨迹、128×128、无评估、无模型保存。它执行真实模型的前向、反向、优化器更新，不更换小模型或改成 LoRA。`DEBUG_STEPS` 表示 optimizer step。可覆盖 `data.train_datasets=[rbm-1m-id]` 测完整数据加载。

要短程检查正式训练的输入规模，显式覆盖调试默认值（先确认所选 GPU 空闲）：

```bash
GPUS_PER_NODE=8 DEBUG_STEPS=3 bash run_scripts/debug_b200.sh 27b \
  training.per_device_train_batch_size=16 data.max_frames=8 \
  data.resized_height=null data.resized_width=null \
  data.dataloader_num_workers=8 data.dataloader_persistent_workers=true \
  'data.train_datasets=[rbm-1m-id]'
```

该命令仍关闭评估、上传和保存。少量步数只能检查抽到的样本，无法保证所有分辨率和轨迹组合都不超显存。

调试结束后，每个 rank 写入 `logs/<RUN_NAME>/memory_rank_<rank>.json`，包含 PyTorch peak allocated/reserved GiB。它不包含所有 CUDA/NCCL 开销；可另用 `nvidia-smi` 观察整卡占用。调试脚本末尾强制覆盖上传与保存设置，避免命令行误开启。

设置环境变量 `ROBOMETER_PROFILE_TIMING=1` 可额外记录每个 rank 的 `timing_rank_<rank>.jsonl`，包含 `time/get_batch`、前向、训练步骤耗时及输入 token 张量形状。`time/get_batch` 是从准备好的迭代器取 batch 的墙钟时间，可能包含预取和设备传输，不能直接等同于磁盘 I/O；训练步骤包含前后向及分布式等待，不包含后续 optimizer.step。计时不额外同步 CUDA，供定位数据等待和各 rank 差异使用，不是纯 GPU kernel profiler。

快速排查 rank 等待可以运行 `bash run_scripts/diagnose_b200_ranks.sh`。默认使用全部 8 卡、27B、每卡 batch 16、8 帧、尺寸 null，只加载小数据子集，重复同一批输入，跑 7 步；第 1–2 步正常自动 GC，第 3–5 步临时关闭自动 GC，第 6–7 步重新启用。此脚本仅用于诊断，不上传、不保存，不能用于正式训练。输入在各 rank 内固定，但各 rank 仍使用各自的 batch。输出 `gc_comparison_rank_*.jsonl` 包含 CUDA 同步后的完整 optimizer step 耗时；首步含编译、优化器初始化等冷启动开销，不能作为稳态基准。

`ROBOMETER_PROFILE_GC=1` 单独启用轻量 GC 回调，记录各次较长回收的墙钟时间、执行线程 CPU 时间及进程 minor page faults，写入 `profile/gc_rank_*.jsonl`。默认不进行定时线程转储；`ROBOMETER_PROFILE_THREADS=1` 可额外启用，但一次开启该功能的诊断在反向传播期间发生 SIGSEGV，原因尚未确认，当前对照不使用它。

若需要追到 C/C++ 调用栈，调试模式支持 `ROBOMETER_PROFILE_NATIVE=1 PY_SPY_BIN=/path/to/py-spy`，让选中的 rank 作为 py-spy 子进程启动。默认只采 rank 0，`ROBOMETER_PROFILE_NATIVE_RANKS=0,1` 可选择多个 rank，输出默认在 `logs/native_profile/`（可用 `ROBOMETER_NATIVE_PROFILE_DIR` 指定）。这是有采样开销的定位工具，不用于正式训练或精确吞吐比较。

原生采样默认 5 Hz，可通过 `ROBOMETER_PROFILE_NATIVE_RATE` 调整。此前 25 Hz 的全量诊断记录到了 `randperm_out_cpu` 和 `recursive_to_list` 的大量样本，4 个训练 step 都完成并生成 trace；但 py-spy 在子进程退出后报 `No child process (os error 10)`，使整个 launcher 退出码为 1，不能把该轮标记为正常退出的训练验证。此次修复的最终验证关闭原生采样。

2026-09-29 的小数据 A/B/A：8 卡、27B、FLA+SDPA、每卡 batch 16、8 帧、尺寸 null、worker 8；rank 0 的输入为 `[16,1293]`。同批次 7 步正常退出；第 2 步正常 GC 为 9.48 秒，关闭自动 GC 的第 3–5 步为 7.38–7.43 秒，恢复 GC 后第 6–7 步为 9.66/10.42 秒（均含 optimizer/CUDA 同步，首步 44.82 秒不计入稳态）。该测试没有复现全量数据下十几秒一次的 GC 停顿，因此不是对全量训练问题已解决的证明。记录在 `logs/qwen27b_bs16_frames8_native_8gpu_w8_gc_aba_20260929_044526/`。

进一步诊断 GPU 低功率等待时，可设置 `ROBOMETER_PROFILE_STEP=2`，只在第 2 个 optimizer step 的前后向内捕获每个 rank 的 CPU/CUDA trace、算子统计及每秒 Python 调用栈，输出到训练目录的 `profile/`。此工具会增加运行时间和主存开销，不应拿开启 profiler 的耗时做吞吐基准；默认关闭。梯度累积为 1 时，每个指定 step 只捕获一次。

仅检查或查看命令：

```bash
bash run_scripts/train_single_b200.sh 27b --check-only
bash run_scripts/debug_b200.sh 397b --check-only
RANK=0 MASTER_ADDR=10.0.0.1 RUN_NAME=preview \
  bash run_scripts/train_multi_b200.sh 397b --dry-run
```

`--dry-run` 不检查权重完整性、不读 key、不启动进程；`--check-only` 执行本地检查，不启动训练或 W&B。

## 输出与恢复

每 500 step 触发评估；27B 和 397B 均每 1,000 step 保存训练 checkpoint。`policy_ranking_samples/step_501/*.json` 保存的是评估样例，不是模型权重；当前评估代码以 global_step + 1 命名该目录。2026-09-29 的首次五机评估在混淆矩阵采样器中失败：旧代码无条件加载 `sentence-transformers/all-MiniLM-L12-v2`，离线缓存缺失会报 LocalEntryNotFoundError，即使 OOD 数据已成功从本地读取。现仅在 `data.load_embeddings=true` 时加载这个辅助模型；Qwen 的 false 配置直接使用任务文本和图片，不需要下载 MiniLM，不改变任务配对或评估项目。已离线验证出错的 USC Trossen 子集：21 条过滤后轨迹，30 个混淆矩阵样例全部成功读取，每个 8 帧；未运行完整多机模型评估。需要 embedding 的旧模型路径仍需预先准备 MiniLM 缓存。

27B 在 `qwen27b.yaml` 中覆盖 `training.max_steps=5000`、`training.save_steps=1000`、`training.save_total_limit=5`。默认新训完整保留 `logs/<RUN_NAME>/checkpoint-{1000,2000,3000,4000,5000}` 五份权重，不按旧 Spearman 指标挑选 checkpoint。397B 在 `qwen397b.yaml` 中覆盖 `training.max_steps=2000`、`training.save_steps=1000`、`training.save_total_limit=3`，新训保存 `checkpoint-1000`、`checkpoint-2000` 两份。397B 的 cosine 调度随总步数调整：前 200 步（10%）warmup 到 `5e-6`，随后在第 2000 步衰减到 0。训练结束也保存当前 step；到达正常终点时写入已有的同一步目录，不额外增加一份 checkpoint。

每份 checkpoint 包含 FSDP 分片模型、优化器状态和 Trainer 恢复状态，可继续训练。多机必须将 `logs/<RUN_NAME>` 放在各节点共同可写的共享存储。分片 checkpoint 需要另作完整推理导出，不能直接将其传给 `from_pretrained()`。

```bash
# 恢复被中断的五机 003 作业；所有节点传相同 RUN_NAME 和共享 checkpoint 路径。
RUN_NAME=rbm_qwen27b_5nodes_003 bash /mnt/liuzihao/project/robometer/run_scripts/train_multi_b200.sh 27b \
  training.resume_from_checkpoint=/mnt/liuzihao/project/robometer/logs/rbm_qwen27b_5nodes_003/checkpoint-2000
```

未指定 resume 时不覆盖已有同名实验。`training.resume_from_checkpoint` 交给 Trainer/FSDP 恢复模型、优化器和 step；上例从 step 2,000 继续到总计 5,000 步，不是额外训练 5,000 步。`training.load_from_checkpoint` 则是原有的仅权重加载路径，两者不要混用。

推理导出可先在一台内存足够的主机合并模型分片，以下命令仅合并权重，不删除训练 checkpoint：

```bash
uv run --no-sync accelerate merge-weights \
  logs/rbm_qwen27b_5nodes_003/checkpoint-5000/pytorch_model_fsdp_0 \
  logs/rbm_qwen27b_5nodes_003/export-5000
```

合并得到 `model.safetensors`；完整 Robometer 推理目录还需正确的模型配置、processor 和训练配置。只有权重合并结果时，尚不能保证 `from_pretrained()` 可直接加载，需完成并验证推理导出。

架构接口参考：[Transformers Qwen3.5 文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)。

## 本机验证记录

27B 完整预训练权重，GPU 2–5，FSDP FULL_SHARD，全参语言主干训练，MetaWorld 真数据，batch=1/卡，2 帧/轨迹、128×128，已完成 2 个 optimizer step。loss 分别为约 2.599、2.598，训练阶段约 9.08 秒（不含加载）。各卡 PyTorch 峰值 allocated 为 101.96–102.33 GiB，reserved 为 106.03–106.44 GiB。W&B disabled，无 checkpoint 保存。对应本地输出：`logs/qwen27b_debug_128px_final/`。

13 项回归测试通过，包括 dense/MoE 权重精确加载、图像前向/反向、三头梯度、下载截断检测、数据目录迁移、图像分辨率生效及梯度累积。另用临时小模型走真实两卡训练入口，成功保存 FSDP 分片模型/优化器，并从 step 1 恢复到 step 2。

上述小输入测试不覆盖正式 8 帧输入；后续补测见下文。长期训练收敛、完整 397B 权重、八机 NCCL、完整大模型 checkpoint 恢复与 OOD 评估仍未验证。

2026-09-29 补测：27B、GPU 0–6（GPU 7 有其他任务）、每卡 batch=16、8 帧/轨迹、尺寸为 null、完整 RBM-1M 训练集合、DataLoader worker=0，完成 3 个 optimizer step，无 OOM。PyTorch peak allocated 为 103.32–104.37 GiB/卡，reserved 为 120.14–121.60 GiB/卡；每 2 秒采样的 nvidia-smi 整卡峰值为 122.04–123.50 GiB/卡。数据初始化约 357.7 秒，Trainer 报告训练耗时 552.2 秒。此记录是七卡短测，不是八卡实测，也不代表 worker=8 的吞吐；三步不能覆盖所有样本组合。原始日志和监控在 `logs/qwen27b_bs16_frames8_native_7gpu_20260929_025910_monitor/`，PyTorch 各卡统计在同名前缀的训练目录。

同日 worker=8 对照：相同七卡、batch 和输入规格，完成 3 步，无 OOM。初始化约 357.3 秒，Trainer 训练耗时约 462.6 秒；PyTorch peak allocated 为 103.42–104.20 GiB/卡，reserved 为 120.06–120.92 GiB/卡，nvidia-smi 采样峰值为 121.45–122.83 GiB/卡。第 2 步所有 rank 的取 batch 等待为 0.025–0.034 秒，训练步骤为 136.00–136.59 秒；第 3 步分别为 0.010–0.015 秒和 147.67–147.87 秒。预取后数据等待很低，仍需诊断前后向和分布式同步。随机样本随 worker 数改变，不能将约 16% 的总耗时下降全部归因于 worker。日志在 `logs/qwen27b_bs16_frames8_native_7gpu_w8_20260929_031742_monitor/`，同名前缀训练目录下有每个 rank 的计时文件。

同日对 worker=8 的第 2 步做逐卡 CPU/CUDA profile：约 175.1 秒的 GPU kernel 时间跨度内，各卡非 NCCL kernel 活跃时间并集为 20.65–24.44 秒，只有 NCCL kernel 活跃的时间为 102.62–109.89 秒。NCCL 活跃时间包含等待其他 rank，不能据此判定物理链路带宽不足。Python 栈反复停在视觉分段长度的 `lengths.tolist()` 和线性注意力参考实现的 CPU 零张量搬运；取 batch 约 0.03 秒。完整 trace、算子表和分析在 `logs/qwen27b_bs16_frames8_native_7gpu_w8_profile_20260929_033655/profile/`。Profiler 有明显开销，导出/汇总耗时较长，不能把这轮总耗时作为吞吐基准。

FLA 补测（同日）：同样七卡、每卡 batch=16、8 帧、尺寸 null、worker=8，完成 3 个 optimizer step，无 OOM，退出码 0。PyTorch peak allocated 为 83.57–84.36 GiB/卡，reserved 为 101.97–103.85 GiB/卡；nvidia-smi 采样峰值 104.22–105.77 GiB/卡。相比原生实现，每卡 allocated 减少约 20 GiB。第 2/3 步前向分别约 81.8/64.4 秒，前后向分别为 141.29–141.54 / 128.91–130.89 秒，取 batch 约 0.025–0.028 / 0.009–0.010 秒。Trainer 三步总训练时间 460.5 秒，与此前 462.6 秒基本持平，不能宣称端到端吞吐已显著提高。短程变长输入可能含首次编译开销，且低功率分布式等待仍可见。运行目录为 `logs/qwen27b_bs16_frames8_native_7gpu_w8_fla_20260929_040451/`，监控位于同名加 `_monitor` 的目录。

在上述运行结束后，另对固定 `[batch=16, seq=1948, value_heads=48, head_dim=128]` 的 Gated Delta Rule 核心做单卡对照，包含反向、同步计时，首轮预热后测 3 轮：原生平均 0.3407 秒，FLA 平均 0.01408 秒，约 24.2 倍。此结果只覆盖单个线性注意力核心，不含投影、卷积、视觉编码器、FSDP 或优化器，不能外推为整步提速 24 倍。原始结果在上述训练目录的 `linear_kernel_benchmark.jsonl`。

本轮新增验证：dense/MoE 小模型 CUDA 多模态前后向与梯度对照通过；48/64 value heads 的线性注意力输出与梯度对照通过；CPU 权重加载/三头反向、梯度累积测试通过。397B 的 94 个权重分片已通过启动前完整性检查，内核及数据检查通过；没有执行单机 397B 全模型训练。
