# USC xArm progress curves

Run from the repository root with the existing environment:

```bash
.venv/bin/python eval_figure/plot_xarm_progress.py
```

Models run sequentially: official Robometer-4B, 27B step 3000, trained 4B step 12000, and 397B step 1000. The first three use GPU 0; 397B uses the existing automatic multi-GPU checkpoint loader with GPUs 0–7 visible. Default batch size is 4 independent 8-frame inputs. This script does not train or modify checkpoints.

The local USC xArm dataset contains **36 trajectories**, each with **32 RGB frames**. Each trajectory produces five independent model inputs:

- Four dense segments: frames 1–8, 9–16, 17–24, 25–32. Each segment starts a fresh context. All eight outputs are retained. Segments are not connected; open circles mark both endpoints.
- One global sample: frames **1, 5, 10, 14, 19, 23, 28, 32**. All eight outputs are retained and connected.

Predictions are raw continuous progress converted from the model's discrete bins by the existing `compute_batched_progress` API. No smoothing, rescaling, forced monotonicity, or linear ground-truth curve is applied. A boundary jump mixes the change of image with the reset of visual context; it is not by itself evidence of a real task regression.

Outputs are under `eval_figure/output/usc_xarm/<model>/`:

- `<trajectory-ID>.png`: two-panel figure, common axes across models.
- `raw/<trajectory-ID>.json`: all 40 predictions, exact frame indices, task, quality label, checkpoint, frame-content hash, and protocol hash.
- `predictions.csv`: long-form predictions for all trajectories.
- `boundary_jumps.csv`: signed differences for 8→9, 16→17, and 24→25.
- `index.html`: gallery grouped by task and quality; root `index.html` links all model galleries.
- `protocol.json`, `checkpoint_load_report.json`, `invocation.json`, `run.log`, `COMPLETE.json`: reproducibility and execution records.

Replot without loading any model or using GPUs:

```bash
.venv/bin/python eval_figure/plot_xarm_progress.py --plot-only
```

Select models with `--models official4b,qwen27b_step3000,trained4b_step12000,qwen397b_step1000`. For a small trial, use `--models official4b --limit 2 --output eval_figure/output/smoke`. Other datasets must use the same saved schema and 32×240×320×3 uint8 frame arrays. The default script processes all trajectories, without random subsampling.

Completed per-trajectory predictions are reused on restart, and their protocol hashes are checked. Use a new `--output` directory if changing the dataset, model manifest, batch size, or trajectory limit.

## ICL 4B: fixed-reference drawer evaluation

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python eval_figure/plot_icl_drawer_progress.py --batch-size 8
```

Uses the completed ICL checkpoint `logs/icl_4b_001/checkpoint-15000`, strictly loading all backbone and ICL-head weights. Uses the saved checkpoint processor and the training conversation format, FP32 parameters with BF16 autocast, and the training image sizing. No original RBM progress/preference/success heads are used.

For the six `Close the drawer` trajectories in the USC xArm subset, each of the 32 frames is an independent query. Every query receives five fixed reference images from trajectory `1d035ec5-5127-4af9-9cca-fe214830d670`: zero-based indices `[0, 17, 22, 25, 31]`, assigned progress `[0, .25, .5, .75, 1]`. There is no query history. The five valid anchor logits are normalized with softmax; padded anchor probabilities are zero. The displayed progress is `sum(probability * assigned_anchor_progress)`, not argmax or a frame-time fraction.

The reference trajectory is also queried and explicitly marked on the plot. Its five reference frames are included among its queries; those are self-reference checks, not independent held-out examples. The other five trajectories test reference transfer. The training sampler used references and query from the same successful trajectory; this evaluation intentionally adds cross-trajectory, failure, and suboptimal queries. The assigned scale is the requested reference-based scale, without claiming per-frame semantic ground truth for the other trajectories.

Output: `eval_figure/output/icl_4b_step15000_close_drawer/`:

- `close_drawer_icl_progress.png` and `.pdf`: six progress curves in one panel, with the five reference images below.
- `predictions.json`: raw five-class logits, probabilities, weighted progress, IDs, frame indices, image hashes, checkpoint and training state.
- `predictions.csv`: all 192 queries and their probabilities.
- `checkpoint_load_report.json` and `COMPLETE.json`: strict weight loading, exact training-collator input equivalence, probability checks, and first-query single/batch comparison.

Replot without inference:

```bash
.venv/bin/python eval_figure/plot_icl_drawer_progress.py --plot-only
```

The script refuses to overwrite existing predictions. To evaluate another ICL 4B checkpoint or batch size, choose `--checkpoint`, `--batch-size`, and a new `--output` directory. `--plot-only` reuses saved predictions and verifies the image-content hashes.

Run the same ICL drawer test for **all saved checkpoints** sequentially on one GPU:

```bash
.venv/bin/python eval_figure/run_icl_drawer_checkpoints.py --gpu 0 --batch-size 8
```

The runner discovers checkpoints in `logs/icl_4b_001`, orders them numerically, validates checkpoint completeness, and reuses compatible cached predictions. Each checkpoint writes to `output/icl_4b_step<step>_close_drawer/`. Cached runs are replotted so reference keyframe markers stay up to date. Shared image hashes, trajectory identities, frame indices, five-class probabilities, and weighted progress are audited across checkpoints.

The gallery is `output/icl_4b_all_checkpoints_close_drawer/index.html`, with links to each PNG, PDF, and prediction CSV. `endpoint_progress.csv` lists the six final-frame predictions at each training step; `validation.json` records the audit and reused results. Override `--run` or `--output-root` when evaluating another run, using a separate output root for a different model run.

## ICL 27B checkpoints

The same drawer protocol also supports ICL 27B FSDP DCP checkpoints:

```bash
.venv/bin/python eval_figure/run_icl_drawer_checkpoints.py \
  --run logs/icl_27b_32gpu_001 --gpu 0 --batch-size 8
```

This run has steps 1000, 2000, 3000, 4000, and 5000. Outputs use `output/icl_27b_step<step>_close_drawer/`; the gallery is `output/icl_27b_all_checkpoints_close_drawer/index.html`. The reference ID, five zero-based keyframe indices, all six query trajectories, weighted-progress formula, and keyframe markers are identical to the 4B test.

`icl_checkpoint_loader.py` constructs the ICL backbone and classification head on meta, checks every FSDP checkpoint tensor name, shape and FP32 dtype, loads all DCP weights into CPU memory, and moves the complete model to one GPU. It uses the training FLA backend, SDPA attention and BF16 autocast, without writing merged weights to disk. This FP32-parameter inference setup requires a large-memory GPU (tested here on B200 180 GB); it is not a two-small-GPU sharded inference entry point. The 4B safetensors loading behavior is retained.

To replot a 27B result without loading weights, specify its output directory:

```bash
.venv/bin/python eval_figure/plot_icl_drawer_progress.py --plot-only \
  --output eval_figure/output/icl_27b_step5000_close_drawer
```
