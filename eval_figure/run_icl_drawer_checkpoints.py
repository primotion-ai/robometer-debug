#!/usr/bin/env python3
"""Run fixed-reference drawer plots sequentially for every saved ICL 4B/27B checkpoint."""
import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval_figure.plot_icl_drawer_progress import REFERENCE_ID, KEYFRAMES, ANCHORS, SUBSET, TASK


def audit(result, checkpoint, step, batch_size):
    assert result['checkpoint'] == str(checkpoint.resolve())
    assert result['training_state']['updates'] == step
    assert result['reference_id'] == REFERENCE_ID and result['reference_frame_indices'] == KEYFRAMES
    assert result['anchor_progress'] == ANCHORS and result['index_base'] == 0
    assert result['dataset'] == SUBSET and result['task'] == TASK
    assert result['batch_size'] == batch_size and result['query_history'] is False
    assert result['batch_vs_single_check']['passed']
    trajectories = result['trajectories']
    assert len(trajectories) == 6 and len({r['id'] for r in trajectories}) == 6
    for row in trajectories:
        assert row['frame_indices'] == list(range(32))
        assert len(row['progress']) == len(row['probabilities']) == len(row['logits']) == 32
        for value, probs, logits in zip(row['progress'], row['probabilities'], row['logits']):
            assert len(probs) == len(logits) == 5
            assert all(math.isfinite(x) for x in [value, *probs, *logits])
            assert all(0 <= p <= 1 for p in probs) and 0 <= value <= 1
            assert abs(sum(probs)-1) < 1e-6
            assert abs(sum(p*a for p,a in zip(probs, ANCHORS))-value) < 1e-6
            exp = [math.exp(x-max(logits)) for x in logits]
            assert max(abs(p-e/sum(exp)) for p,e in zip(probs,exp)) < 1e-6
    return sorted((r['id'],r['quality_label'],r['task'],result['frame_sha256'][r['id']]) for r in trajectories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'logs/icl_4b_001')
    parser.add_argument('--output-root', type=Path, default=ROOT/'eval_figure/output')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    checkpoints = sorted([p for p in args.run.glob('checkpoint-*') if p.name.split('-')[-1].isdigit()],
                         key=lambda p:int(p.name.split('-')[-1]))
    if not checkpoints:
        raise ValueError('No checkpoints found')
    model_name = json.loads((checkpoints[0]/'icl_state.json').read_text())['model']
    if model_name not in ('4b', '27b'):
        raise ValueError('Only ICL 4B/27B runs are supported')
    model_label = model_name.upper()
    for checkpoint in checkpoints:
        if json.loads((checkpoint/'icl_state.json').read_text())['model'] != model_name:
            raise ValueError('Mixed model types in run')
        if not (checkpoint/'model.safetensors').is_file() and not (checkpoint/'pytorch_model_fsdp_0/.metadata').is_file():
            raise ValueError(f'No model weights: {checkpoint}')
        for filename in ['icl_config.yaml','icl_state.json','processor']:
            if not (checkpoint/filename).exists():
                raise ValueError(f'Incomplete checkpoint: {checkpoint/filename}')
    overview = args.output_root/f'icl_{model_name}_all_checkpoints_close_drawer'
    overview.mkdir(parents=True, exist_ok=True)
    results, identity = [], None
    for checkpoint in checkpoints:
        step = int(checkpoint.name.split('-')[-1])
        dest = args.output_root/f'icl_{model_name}_step{step}_close_drawer'
        dest.mkdir(parents=True, exist_ok=True)
        reuse = (dest/'predictions.json').exists()
        if reuse:
            audit(json.loads((dest/'predictions.json').read_text()), checkpoint, step, args.batch_size)
        command = [sys.executable, str(ROOT/'eval_figure/plot_icl_drawer_progress.py'),
                   '--checkpoint',str(checkpoint.resolve()),'--output',str(dest.resolve()),
                   '--batch-size',str(args.batch_size)]
        if reuse:
            command.append('--plot-only')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
        print(f'START step {step}: {"replot cached predictions" if reuse else "run 192 queries"}',flush=True)
        started=time.time()
        (dest/'all_checkpoints_invocation.json').write_text(json.dumps(dict(command=command, gpu=args.gpu,
            reused_predictions=reuse, started=started),indent=2))
        with (dest/'all_checkpoints_run.log').open('a') as stream:
            subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result=json.loads((dest/'predictions.json').read_text())
        current=audit(result,checkpoint,step,args.batch_size)
        if identity is None:
            identity=current
        else:
            assert current == identity, 'Different input trajectories across checkpoints'
        report=json.loads((dest/'checkpoint_load_report.json').read_text())
        assert report['missing_keys']==report['unexpected_keys']==[]
        assert report['collator_exact_training_equivalence']
        for extension in ['png','pdf']:
            assert (dest/f'close_drawer_icl_progress.{extension}').stat().st_size > 0
        results.append((step,dest,result,reuse))
        print(f'DONE step {step}: {time.time()-started:.1f}s',flush=True)
    with (overview/'endpoint_progress.csv').open('w') as stream:
        writer=csv.writer(stream)
        writer.writerow(['checkpoint_step','trajectory_id','quality_label','is_reference','final_progress'])
        for step,_,result,_ in results:
            for row in result['trajectories']:
                writer.writerow([step,row['id'],row['quality_label'],row['id']==REFERENCE_ID,row['progress'][-1]])
    cards=[]
    for step,dest,_,reuse in results:
        relative=os.path.relpath(dest,overview)
        cards.append(f'<section><h2>ICL {model_label} / step {step}</h2><p>'
            f'<a href="{relative}/close_drawer_icl_progress.png">PNG</a> · '
            f'<a href="{relative}/close_drawer_icl_progress.pdf">PDF</a> · '
            f'<a href="{relative}/predictions.csv">CSV</a></p>'
            f'<img loading="lazy" src="{relative}/close_drawer_icl_progress.png" alt="Step {step}"></section>')
    (overview/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>ICL {model_label}: all checkpoints</title>'
        '<style>body{font-family:system-ui;max-width:1500px;margin:24px auto;padding:0 20px}img{width:100%}section{margin:40px 0}</style>'
        f'<h1>ICL {model_label} — Close the drawer — all checkpoints</h1>'
        '<p>Identical six trajectories and five reference keyframes [0, 17, 22, 25, 31]. '
        'Each frame is queried independently; yellow diamonds mark reference keyframes on the predicted curve.</p>' + ''.join(cards))
    (overview/'validation.json').write_text(json.dumps(dict(model=model_name, checkpoint_steps=[s for s,_,_,_ in results],
        trajectories_per_checkpoint=6,queries_per_checkpoint=192,total_queries=len(results)*192,
        input_images_and_metadata_identical=True,probabilities_and_weighted_progress_verified=True,
        strict_weight_loading_verified=True,reused_prediction_steps=[s for s,_,_,reuse in results if reuse]),indent=2))
    provenance=overview/'provenance';provenance.mkdir(exist_ok=True)
    hashes={}
    for path in [Path(__file__),ROOT/'eval_figure/plot_icl_drawer_progress.py',ROOT/'eval_figure/icl_checkpoint_loader.py']:
        (provenance/path.name).write_bytes(path.read_bytes())
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (overview/'source_sha256.json').write_text(json.dumps(hashes,indent=2))
    print(f'COMPLETE: {overview}',flush=True)


if __name__ == '__main__':
    main()
