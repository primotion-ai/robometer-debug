# ICL reference-conditioned progress

Independent training path on branch `codex/icl-progress`. It does not instantiate
RBM heads or use preference, success, rewind, or the original progress-label
pipeline. Existing Robometer entrypoints are unchanged.

## Objective and inputs

Each sample contains a task instruction, K ordered reference images (2–11), and
one query image from the same successful trajectory. References can start/end
anywhere in the trajectory. Query time is strictly inside the reference span and
is never one of the reference frame indices. Index inequality does not guarantee
visually distinct pictures in a stationary scene.

Reference j defines progress j/(K-1). Between reference times ti and ti+1, alpha =
(t-ti)/(ti+1-ti). The target distribution assigns 1-alpha and alpha to those two
anchors, zero elsewhere. Only soft-target cross-entropy is optimized. Expected
progress is the probability-weighted sum of the anchor progress values.

A shared scorer compares the query representation (after the query image) with
each reference representation. Its output length follows K; padded anchors are
excluded before softmax. Text explicitly labels the references and query; standard
Qwen image boundary tokens delimit images. No new tokenizer tokens are needed.
Timestamps, frame indices, and target progress are never supplied as model inputs.

## Data and batches

Default input is the deduplicated `rbm-1m-id` training list. Only `successful`
rows are eligible. Conflicting partial scores below 1, and trajectories shorter
than min_keyframes+1, are excluded. No files are removed or rewritten.

One GPU batch first samples B+1 candidates, retaining the first B valid samples.
If the entire internal non-keyframe set is empty, the candidate is rejected and
additional candidates are sampled. An empty individual interval does not invalidate
other intervals. Refilling is bounded (100 rounds). Metadata sampling precedes
image decoding. Final per-GPU batch remains B, including under DDP/FSDP.

By default, cache subsets are selected uniformly, then eligible trajectories
uniformly within the selected subset. Set data.source_balanced=false to sample
proportional to eligible trajectory counts instead. This is sampling with replacement;
raw dataset counts are not batch mixture ratios.

A stable hash of (dataset key, trajectory ID, split seed) holds out 1% of whole
trajectories for validation before frame sampling. No fallback moves held-out
trajectories into training. This is a same-trajectory ICL evaluation, not a test of
cross-trajectory reference transfer or the old RBM OOD policy-ranking benchmark.
Different cache rows/views may still correspond to related physical episodes.

Useful overrides:

- `data.exclude_datasets=[full_cache_key,...]`
- `data.exclude_sources=[data_source,...]`
- `data.min_keyframes=2 data.max_keyframes=11`
- `data.validation_fraction=0.01`
- `data.workers=10` (per GPU; 80 training workers per eight-GPU node)
- `data.max_pixels=null` (native processor sizing; an explicit cap is optional)
- `data.max_trajectories_per_source=null` (set a small number only for smoke tests)

Each run writes config.yaml, dataset_summary.json and metrics.jsonl. Default
whole-cache scan: 146,294 non-success rows + 601 short successful rows removed;
903,405 train / 9,070 validation rows. Data or split overrides change these numbers.

## Presets

| Model | Parallelism | Precision | Per-GPU batch | LR | Steps | Save |
|---|---|---|---:|---:|---:|---:|
| 4b | DDP | BF16 compute / FP32 training states | 32 | 2e-5 | 15,000 | 3,000 |
| 27b | FSDP FULL_SHARD | BF16 compute / FP32 training states | 16 | 1e-5 | 5,000 | 1,000 |
| 397b | FSDP FULL_SHARD | FP16 parameters, gradients, AdamW moments | 4 | 5e-6 | 2,000 | 1,000 |

Per-GPU batch sizes and LRs match the current Robometer qwen4b/qwen27b/qwen397b
recipes (including b200_base inheritance). These larger ICL batches have not yet
been GPU-tested; the validation below used smaller batches. Each ICL sample can
contain up to 12 images, so equal batch counts do not imply equal memory usage.
Accumulation is
1; cosine LR reaches zero at max_steps, with 10% warmup. Vision is frozen. 4B/27B
use non-reentrant Transformers gradient checkpointing; 397B uses FSDP activation
checkpointing on all language/vision blocks and AdamW eps=1e-4. 27B/397B reuse FLA;
397B also reuses the existing eager-expert backward and shared-expert FP16 fixes.
The 397B full training path still requires real multi-node validation.

No extra loss or loss-threshold rejection is used. Nonfinite loss/gradient windows
are rejected on all ranks after backward, without optimizer/scheduler advancement;
only successful updates count toward max_steps. Rejection count is bounded and
logged. This does not guarantee the numerical stability of pure FP16 training.

## Launch

From the repository root, with the existing mounted uv `.venv`:

```bash
# Full single-node training, eight GPUs by default
RUN_NAME=icl_4b_001 bash run_scripts/icl/single.sh 4b
RUN_NAME=icl_27b_001 bash run_scripts/icl/single.sh 27b

# Short debug, W&B/evaluation/checkpoint saving forcibly disabled
RUN_NAME=icl_4b_debug bash run_scripts/icl/debug.sh 4b
RUN_NAME=icl_27b_debug bash run_scripts/icl/debug.sh 27b

# Fast data setup for smoke tests, retaining real cached images
RUN_NAME=icl_4b_small bash run_scripts/icl/debug.sh 4b \
  'data.datasets=[aliangdw_metaworld_metaworld_train]' data.min_keyframes=11

# Inspect configuration/command without keys, model loading or training
bash run_scripts/icl/single.sh 27b --dry-run
```

Multi-node: platform executes the command once per container/node, with matching
RUN_NAME, MASTER_ADDR and MASTER_PORT. The 27B default is four nodes: NNODES=4,
NPROC_PER_NODE=8, WORLD_SIZE=32, NODE_RANK=0..3. Each GPU uses batch 16 and 10
training workers; global batch is 512 with accumulation=1. The 397B default remains
five nodes: NNODES=5, NPROC_PER_NODE=8, WORLD_SIZE=40, NODE_RANK=0..4.
Explicit platform NNODES overrides the default; configure the actual allocation
consistently. NPROC_PER_NODE=auto is supported.

```bash
RUN_NAME=icl_27b_32gpu_001 bash /mnt/liuzihao/project/robometer/run_scripts/icl/multi.sh 27b
RUN_NAME=icl_397b_40gpu_001 bash /mnt/liuzihao/project/robometer/run_scripts/icl/multi.sh 397b
# Same launcher supports 27b (and 4b DDP) on multiple nodes.
```

All scripts disable HTTP/HTTPS/ALL/W&B proxy environment variables and set
NO_PROXY=*. Formal training uses W&B on global rank 0 only, reading the ignored
.secrets/wandb_api_key or WANDB_API_KEY_FILE. Debug never reads a key. Use
logging.wandb=false in the normal launcher for offline validation runs.

Single-node 397B retains full-parameter training with CPU offload by default,
not a claim that the node has enough memory. CPU offload requires accumulation=1.

## Checkpoints and resume

`checkpoint-N` is an Accelerate training-state checkpoint (including the ICL head,
optimizer, scheduler, RNG and successful-update/data cursor), not an original RBM
checkpoint or a standalone Hugging Face model export. Under FSDP it is sharded.
The checkpoint also contains icl_config.yaml, processor/ and icl_state.json.
A final checkpoint is saved if the last step was not already a scheduled save.
There is no automatic checkpoint deletion.

Resume requires the same model/data config, seed, max_steps, world size, local
batch and accumulation. Use the same overrides that defined the original run:

```bash
RUN_NAME=icl_27b_001 bash run_scripts/icl/single.sh 27b \
  train.resume=/absolute/path/to/logs/icl_27b_001/checkpoint-1000
```

Sampler indices resume directly at the consumed-batch cursor, so prefetched but
unconsumed candidates are not incorrectly skipped. A new experiment needs a new
RUN_NAME; existing output directories are never deleted automatically.

## Validation

CPU tests: `.venv/bin/python -m unittest discover -s tests/icl -v`.
Run evidence is stored under logs/icl_validation. Real backbone smoke tests use
short runs, local data, no W&B and no full-model checkpoint writes; the small
FSDP integration harness separately verifies checkpoint/resume and evaluation.

Validated on this eight-B200 node (2026-10-08), using the full eligible data pool:

| Backbone | Batch/GPU | References | Successful updates | Peak allocated/GPU |
|---|---:|---|---:|---|
| 4B | 8 | random 2–11 | 3 + evaluation | 79.5 GiB |
| 27B | 4 | fixed 11 (maximum-context smoke) | 3 + evaluation | 54.6–55.1 GiB |

Canonical logs are `4b_fullpool_smoke_002.log` and
`27b_fullpool_smoke_002.log`. Both verify AdamW moment coverage against every
rank's live trainable parameters. Earlier 27B smoke runs are superseded: they
exposed stale optimizer references after FSDP meta-parameter materialization.
The optimizer is now constructed after model wrapping, with runtime checks.
The last two updates took about 1.4–1.6 seconds (4B) and 2.6–3.1 seconds (27B);
these short runs do not establish sustained throughput or worst-case memory.

Eight CPU tests, all nine launcher dry-runs, and real processor checks for all
three backbones passed. A two-GPU tiny FSDP model with nonzero-rank meta loading
tested accumulation=2, sharded checkpoint saving, evaluation, and resume from
update 1 through update 3. Resumed losses matched within 1e-4, not bitwise.
397B passed local architecture/processor/FLA/wrap-class checks only; its weights,
forward/backward, memory capacity and multi-node training remain unvalidated.
