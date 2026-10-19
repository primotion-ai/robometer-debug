# Astra 偏序进度评测

两阶段：各模型先生成每条轨迹 32 个进度值，再在同一份冻结帧对上比较 `lt / eq / gt`。

## 独立预测入口：每模型 8 核与预加载

后续单模型运行使用 `python -m eval_astra_bench.predict`。ICL 4B、27B 和官方 4B 共用这个入口；每个进程自行读取与解码图像，不依赖公共预热进程。

默认配置：

- `--cpu-cores 8`：绑定到 8 个不同物理核中的各一个逻辑 CPU；PyTorch / BLAS 线程数也设为 8。CPU 亲和性限制本进程的位置，不独占机器上的这些核。
- `--io-workers 8`：进程内 8 个 PNG 读取与 RGB 解码线程。
- `--prefetch-batches 2`：在当前批之外提前读取后面 2 批，和 GPU 推理重叠；`0` 关闭提前加载，但保留批内并行读取。ICL 一批为一条轨迹（32 个 query 帧），官方一批为 `--batch-size` 条轨迹；不会把整个数据集预先解码。
- `--cpu-slot`：默认从 `CUDA_VISIBLE_DEVICES` 的第一个数字 GPU 编号选择 CPU 核分组。GPU 0～7 分别使用互不重叠的 8 核分组；使用 GPU UUID、受限 cpuset 或自定义并行方式时显式指定 slot，所有并行进程保持相同 `--cpu-cores`。当前可用核不足时直接报错。
- ICL 的五张参考图在每个进程内缓存，查询帧按 manifest 顺序交付；官方只加载本模式需要的 8 / 16 / 32 帧，保持原插值规则。

例如单独测试一个 ICL 检查点：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m eval_astra_bench.predict \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --adapter icl --checkpoint logs/icl_4b_001/checkpoint-15000 \
  --model-id icl-4b-step15000 --batch-size 32 \
  --cpu-cores 8 --io-workers 8 --prefetch-batches 2 \
  --output eval_astra_bench/outputs/icl4b_step15000_prefetch/predictions.jsonl
```

ICL 27B 只需更换检查点和 model-id。官方 4B 使用 `--adapter official --checkpoint /mnt/base_model/Robometer-4B --base-model /mnt/base_model/Qwen3-VL-4B-Instruct`，搭配 `--official-frames 32 --batch-size 8` 或 `--official-frames 8 --batch-size 32`。

统一输出与 `--resume` 规则沿用原协议；I/O 参数单独追加记录到 `predictions.io.jsonl`，完成后也写入 `predictions.runtime.json`。8 个读取线程和计算线程共同运行在本进程配置的 8 核上。操作系统仍可自然共享文件缓存，但每个进程拥有自己的解码图片和预取队列。

正在执行的 `all_ckpts_20261018` 队列继续使用锁定的旧入口 `python -m eval_astra_bench predict`，避免更改代码指纹使待运行任务被拒绝。新入口的 PNG 并发、预取边界、异常传播、CPU 分组及与旧推理路径的等价性由 `test_prefetch.py` 覆盖。

## 当前冻结协议

`fixtures/ood_v1/benchmark.json` 记录完整配置、数据版本、排除名单和内容哈希。

- 数据来源：`/mnt_slow/wangchunfeng/dataset/robometer-1m-ood-label/current`，创建时固定到实际版本 `2026-10-17-r177-v016-mixed`。
- 仅排除 `source=rfm_new_mit_franka_rfm` 且 `instruction=Fold the towel in half` 的 **56 条**。其他来源的毛巾任务保留。
- 剩余 **514 条轨迹 / 16,448 帧 / 41 种任务指令**。
- 每条轨迹从 496 个不同帧组合中无放回随机采 **32 对**，共 **16,448 个 within 对**。
- 跨轨迹默认按同一 `reference_key` 分组，每组从不同轨迹的全部帧组合中无放回采 **256 对**，共 **11,264 个 cross 对**。只有一条轨迹的组无法跨轨迹配对，记录在 `skipped_cross_groups`。
- 总计 **27,712 对**；随机种子 `20261018`。A/B 的方向也随机化，不保证 A 是较早帧。
- 同一任务存在新旧 Astra 标尺，因此默认跨轨迹配对额外要求标尺相同。可用 `prepare --pair-scope task` 生成按任务文字跨标尺配对的另一个版本；不要混淆两个 benchmark 的得分。
- 默认保留另外 **2 条已知人工／模型终态冲突**，严格按照现有 Astra progress 计分。可用 `--exclude-known-conflicts` 创建另一个版本。不会把人工终态答案写回密集进度。
- ICL 参考轨迹**也参与计分**，包括它的参考帧。报告额外按帧对是否涉及参考轨迹分组。

对于 gold 和 prediction 都使用同一个阈值 `epsilon=0.1`：

```text
A - B >  epsilon : gt
A - B < -epsilon : lt
abs(A - B) <= epsilon : eq
```

两者类别完全相同得 1，否则得 0。主分数为全部帧对的 `correct / total`。另报告 within/cross、任务、gold 类别、参考轨迹曝光、已知冲突分组分数，以及 3×3 混淆矩阵。`task_macro_accuracy`、`mode_macro_accuracy` 是附加指标，不替代按总对数归一化的主分数。

标签中允许进度回退；不强制端点、不平滑、不做每轨迹 min-max。`eq` 是逐对判定的容差关系，不假设它具有传递性。固定集合中 13,674 对为 eq，因此永远输出相同进度的基线得分约 **49.34%**。

## 固定文件与模型输入

```text
fixtures/ood_v1/
  benchmark.json   协议、内容哈希、唯一 benchmark_id
  inputs.jsonl     图像、任务、固定 ICL 参考；不含目标 progress
  targets.jsonl    原始 Astra 进度与标签来源信息
  pairs.jsonl      固定帧对、gold_relation、审计分组标记
```

`inputs.jsonl` 中的图像路径相对于 `benchmark.json` 记录的数据版本目录。`--data-root` 可指向完整副本，必须具有同一来源清单和 checksum 清单。推理适配器只接收输入行，不接收目标标签或帧对答案。

通常直接使用已生成的 `fixtures/ood_v1`。修改阈值、种子、采样预算或筛选规则时新建一个目录：

```bash
cd /mnt/liuzihao/project/robometer
python3 -m eval_astra_bench prepare \
  --output eval_astra_bench/fixtures/ood_v2 \
  --threshold 0.1 --seed 20261018 \
  --within-per-trajectory 32 --cross-per-group 256
```

不会覆盖已有冻结目录。每次读取都会核对三个 JSONL 的内容哈希。所有模型使用同一个 benchmark_id 才能直接比较。

## 统一模型 output

每个模型输出一个 JSONL，每条轨迹一行。必须包含全部 514 个 UUID，每条完整提供 0～31 的进度。字段规范在 `prediction.schema.json`；以下示意省略数组中间项，实际文件必须写全：

```text
schema_version: "astra_progress_predictions/v1"
benchmark_id:   从 benchmark.json 原样复制
trajectory_id:  原始 UUID
frame_indices:  [0, 1, ..., 31]
progress:       [p0, p1, ..., p31]，有限数值，范围 [0,1]
```

可用 `core.prediction_row(info, input_row, progress_list)` 创建一行合法记录。未来其他模型只需生成该格式，不必复用现有模型加载器。一个文件只放一个模型/检查点/推理协议的预测；推荐同时保留 `<name>.run.json`，记录模型名称、检查点与推理配置。内置推理会自动生成这个文件及 `<name>.load.json`。

未知/重复 UUID、缺失轨迹、缺帧、乱序帧号、NaN/Inf、越界进度、不同 benchmark_id 都会报错，不会静默丢弃或当成零分。外部模型若输出未归一化 reward，需由其适配器定义明确、固定的全局标定方式；不要按每条轨迹重新归一化。

## 阶段一：ICL 4B / 27B

每个场景参考组从原始 `source_quality_label=successful` 的轨迹中用固定种子选择一条参考，冻结在 `inputs.jsonl`。目前共 **42 条参考**；两个其他来源的毛巾场景各用自己的参考。只根据原始成功类别选择，不用 Astra 的逐帧分数挑参考。

关键帧严格使用用户指定的 **[0, 8, 16, 24, 31]**，提示中的进度为 **[0, .25, .5, .75, 1]**。四段跨度 8、8、8、7，属于近似均匀采样；这里定义的是参考时间进度，不保证这些帧在 Astra 标尺下也恰好是相同百分比。

每次输入五张参考图和一张 query，不提供 query 历史。直接调用训练模块 `robometer.icl_progress.collator.build_conversation()`，采用同一 chat template、图像预处理、vision_end anchor 位置、padding mask 和 query token 位置。保留原训练 prompt，包括 “not necessarily completion of the entire task” 这句，不擅自改成另一种提示。最终 progress 为五个有效 anchor 的 softmax 概率加权和。参考轨迹自身的预测也由模型实际计算，不直接填入参考百分比。

```bash
# 4B；检查点可换成希望评测的版本
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m eval_astra_bench predict \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --adapter icl --checkpoint logs/icl_4b_001/checkpoint-15000 \
  --model-id icl-4b-step15000 --batch-size 4 \
  --output eval_astra_bench/outputs/icl4b.jsonl

# 27B；支持 FSDP DCP，不需要先合并检查点
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m eval_astra_bench predict \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --adapter icl --checkpoint logs/icl_27b_32gpu_001/checkpoint-1000 \
  --model-id icl-27b-step1000 --batch-size 2 \
  --output eval_astra_bench/outputs/icl27b.jsonl
```

当前用户训练的 397B 没有 ICL，不纳入本轮任务。通用加载器预留的 397B 路径使用同一个 `--adapter icl`，传入对应 ICL 检查点，并用 `--device-map auto` 将模型放到**同一主机的多张可见 GPU**。它读取 `icl_config.yaml` 与 `icl_state.json`，保留训练的纯 FP16 权重，并应用现有 shared-expert FP32 中间计算修复。4B/27B 保持 FP32 参数、BF16 autocast。加载支持单文件/分片 safetensors 和 FSDP DCP，严格检查键、形状及精度。

自动映射要求可见 GPU 总显存足够，加载 DCP 还需要足够主机内存；不自动量化、CPU offload 或启动跨节点任务。397B 的实际 ICL 检查点端到端验证尚未执行，不能把实现支持当作已跑通的实验结果。

## 阶段一：官方 Robometer 4B

已核查官方仓库，而不是假设它只能输出八个值：

- [官方本地推理示例](https://github.com/robometer/robometer/blob/main/scripts/example_inference_local.py) 的 `compute_rewards_per_frame_local()` 把完整输入序列组成一个 `ProgressSample`，一次 forward 后取逐帧结果。
- [官方 collator](https://github.com/robometer/robometer/blob/main/robometer/data/collators/rbm_heads.py) 的 `_add_vision_content_to_list()` 在每张图后插入 `<|prog_token|>`；不是固定八维输出。
- [官方 reward alignment sampler](https://github.com/robometer/robometer/blob/main/robometer/data/samplers/eval/reward_alignment.py) 在 `use_frame_steps=False` 时提供整轨迹路径；[默认配置](https://github.com/robometer/robometer/blob/main/robometer/configs/config.yaml) 中有 `data.max_frames: 16`。
- 本机发布检查点配置为训练 `max_frames=8`，但对同一条真实 OOD 轨迹实测输入 8、16、32 帧，分别返回 8、16、32 个有效进度值。这证明接口支持变长，不代表三种输入长度的准确率相同。官网示例图本身的确切导出设置尚未确认。

因此默认使用**整轨迹 32 帧一次 forward → 32 个直接预测**。不采用逐目标帧重采样历史前缀。使用已有离散进度头的 `convert_bins_to_continuous` 解码，保留训练预处理配置。

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/python -m eval_astra_bench predict \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --adapter official --checkpoint /mnt/base_model/Robometer-4B \
  --base-model /mnt/base_model/Qwen3-VL-4B-Instruct \
  --model-id official-robometer-4b-full32 --official-frames 32 \
  --output eval_astra_bench/outputs/official32.jsonl
```

也支持用户提出的 fallback：`--official-frames 8`，从原始 32 帧按 `[0,4,8,13,17,22,26,31]` 均匀采样，一次得到 8 个值，按**原始存储帧索引**线性插值至 32 点；`--official-frames 16` 同理。输出额外保存 `adapter_details.sampled_frame_indices` 和 `sampled_progress`，区分直接预测与插值结果。8/16/32 三种协议应使用不同 model-id 和输出文件。

## 阶段二：校验与计分

```bash
python3 -m eval_astra_bench validate \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --predictions eval_astra_bench/outputs/icl4b.jsonl

python3 -m eval_astra_bench score \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --predictions eval_astra_bench/outputs/icl4b.jsonl \
  --output eval_astra_bench/outputs/icl4b_scores
```

`metrics.json` 给出主分数和分组统计；`pair_results.jsonl` 保存每对的预测类别、gold 类别、两个预测值和 0/1 得分。阶段二仅依赖 Python 标准库，不需要 GPU 或模型包。

`predict --dry-run` 可先检查配置，不加载模型；`--limit 1` 仅处理一条待预测轨迹做冒烟检查。之后用相同配置和 `--resume` 继续，已有完整轨迹不重算。部分预测不能正式计分。checkpoint、batch size 或推理协议改变会拒绝续写，需另建输出文件；`--limit` 可以改变。

## 验证

```bash
python3 -m unittest eval_astra_bench.test_benchmark -v
.venv/bin/python -m unittest eval_astra_bench.test_model_io eval_astra_bench.test_prefetch -v
```

测试覆盖阈值边界、固定种子复现、排除范围、参考轨迹保留、帧对唯一性和任务/标尺约束、缺失/非法预测拒绝、输入文件篡改拒绝、满分 oracle、常量基线、官方整轨迹输出及插值，以及 ICL 输入与训练 collator 的逐张量一致性。`outputs/validation/` 保存本机实际验证日志，不属于正式模型评测结果。

## 2026-10-18 全检查点输出队列

本轮仅生成统一预测，不调用 `score`。tmux 会话：`astra_outputs_20261018`。

- ICL 4B：3000、6000、9000、12000、15000，query-frame batch size 32。
- ICL 27B：1000、2000、3000、4000、5000，query-frame batch size 32。
- 官方 4B：32 帧直接输出，每批 8 条轨迹；8 帧采样插值，每批 32 条轨迹。
- GPU 0～7 各自运行独立模型，先启动较慢的 27B；每张卡完成后从共享队列领取下一项。
- 输出根目录：`outputs/all_ckpts_20261018/`。每项位于 `<model-id>/predictions.jsonl`。
- `plan.json` 保存完整命令和代码指纹；`status.json` 保存任务状态与最近轨迹数；`logs/` 保存逐任务日志；`queue.log` 保存整体进度。
- 每条轨迹完成即刷盘，任务完成后只做格式和完整性校验。中断后可重新运行 `launch.sh`，使用相同元数据恢复，不重算已完成轨迹。
- 若队列运行过程中修改推理相关代码，尚未启动的任务会拒绝执行，避免同一批实验混用实现。

查看：

```bash
tmux attach -t astra_outputs_20261018
cat /mnt/liuzihao/project/robometer/eval_astra_bench/outputs/all_ckpts_20261018/status.json
```

本轮发现串行读 PNG 会等待 Ceph 元数据请求。可在推理保持运行时启动共享文件缓存预热：

```bash
python3 -u -m eval_astra_bench.warm_image_cache \
  --benchmark eval_astra_bench/fixtures/ood_v1 \
  --queue-status eval_astra_bench/outputs/all_ckpts_20261018/status.json \
  --output eval_astra_bench/outputs/all_ckpts_20261018/cache_warmup.json \
  --workers 8
```

预读按当前最慢任务的进度向后读取，再读取此前轨迹；使用有界线程池，只读原文件，不改动模型、prompt 或输出。文件缓存由同机推理进程共享；实际收益取决于文件系统缓存和内存压力，不保证缓存常驻。当前已放在同一 tmux 会话的 `data_prefetch` 窗口，日志为 `cache_warmup.log`，测量记录为 `io_diagnosis.json`。增加写线程没有依据：预测仅每条轨迹写一行，当前捕获的阻塞发生在 Ceph 数据源。

## 2026-10-19 等权关键帧标注复评

新发布包为 `/mnt_slow/wangchunfeng/dataset/robometer-1m-ood-label/ood-r177-equal-keyframes-v1`（发布说明中的 `/root/hunshannas/...` 在本机映射到该共享路径）。使用 `frame_labels[].progress` 的等权 P 作为目标，不使用 `base_progress`。

本版有 502 条轨迹，排除了全部三个来源的 68 条 `Fold the towel in half`；旧评测只排除 MIT Franka 的 56 条。因此本次比旧评测少 12 条。模型仍使用原先成功轨迹的 `[0,8,16,24,31]` 五帧 ICL，不把新标注模型自己的参考锚点替换为待测模型输入。

复评入口：

```bash
python3 -m eval_astra_bench.rescore_release \
  --data-root /mnt_slow/wangchunfeng/dataset/robometer-1m-ood-label/ood-r177-equal-keyframes-v1 \
  --original-benchmark eval_astra_bench/fixtures/ood_v1 \
  --original-plan eval_astra_bench/outputs/all_ckpts_20261018/plan.json \
  --benchmark-output eval_astra_bench/fixtures/ood_equal_keyframes_v1 \
  --output eval_astra_bench/outputs/equal_keyframes_v1_20261019 \
  --workers 8
```

输出目录必须不存在。该入口先检查新旧任务、逐帧哈希、参考帧和参考标签，再用 8 个读线程校验全部 16,064 张新图片的实际字节；全部一致才复用已有模型预测。新的 benchmark ID 对应新标签，转换后的预测另存，每个模型的 `reuse.json` 保留旧预测路径、SHA256 和原运行元数据。

帧对直接保留旧固定集合的 ID、方向、端点，过滤新集合中不存在的轨迹后共 26,816 对：同轨迹 16,064 对、跨轨迹 10,752 对，阈值仍为 0.1。每个跨轨迹对均检查属于同任务、同一新标注标尺。跨轨迹抽样分层仍继承旧版参考标尺的配额，不按新版 40 个标尺重新随机采样，因此部分新标尺会合并旧版多个抽样分层。各模型使用完全相同的帧对。新版本未提供原版的人工冲突字段，该诊断维度标记为未提供，不能据此宣称没有冲突。

结果保存到 `outputs/equal_keyframes_v1_20261019/summary.csv`、`summary.json` 和各模型的 `scores/`。旧版本 fixture、预测和得分全部保留。
