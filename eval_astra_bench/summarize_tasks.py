"""Aggregate existing pair results by task and mode; does not run inference."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from .core import load_benchmark, read_jsonl, unique_by_id, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    info,inputs=load_benchmark(args.benchmark)
    summary=json.loads((args.results/'summary.json').read_text())
    if summary['benchmark_id'] != info['benchmark_id']:
        raise ValueError('Benchmark mismatch')
    if args.output.exists():
        raise FileExistsError(args.output)
    pairs=unique_by_id(read_jsonl(args.benchmark/'pairs.jsonl'),'pair_id')
    tasks=sorted({r['instruction'] for r in inputs})
    trajectories=Counter(r['instruction'] for r in inputs)
    totals=Counter((p['instruction'],p['mode']) for p in pairs.values())
    rows=[]
    models=[r['model_id'] for r in summary['results']]
    for model,overall in zip(models,summary['results']):
        details=unique_by_id(read_jsonl(args.results/model/'scores/pair_results.jsonl'),'pair_id')
        if details.keys()!=pairs.keys():
            raise ValueError(f'Pair set mismatch: {model}')
        correct=Counter()
        for uid,item in details.items():
            pair=pairs[uid]
            if item['mode']!=pair['mode'] or item['gold_relation']!=pair['gold_relation']:
                raise ValueError(f'Pair metadata mismatch: {model}: {uid}')
            expected=int(item['predicted_relation']==pair['gold_relation'])
            if item['correct']!=expected:
                raise ValueError(f'Incorrect stored score: {model}: {uid}')
            correct[pair['instruction'],pair['mode']]+=expected
        for mode in ('within','cross'):
            assert sum(n for (task,m),n in totals.items() if m==mode)==overall[f'{mode}_total']
            assert sum(n for (task,m),n in correct.items() if m==mode)==overall[f'{mode}_correct']
        for index,task in enumerate(tasks,1):
            row=dict(task_id=f'T{index:02}',task=task,model_id=model,trajectories=trajectories[task])
            for mode in ('within','cross'):
                count=totals[task,mode]
                row.update({f'{mode}_correct':correct[task,mode],f'{mode}_total':count,
                            f'{mode}_accuracy':correct[task,mode]/count if count else None})
            rows.append(row)
    args.output.mkdir(parents=True)
    with (args.output/'per_task.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(args.output/'per_task.json',dict(benchmark_id=info['benchmark_id'],tie_threshold=info['tie_threshold'],
               grouping='exact task instruction; pool frame-pair counts across sources within task',
               missing_cross='null means no cross-trajectory pairs, not zero accuracy',rows=rows))
    lookup={(r['task'],r['model_id']):r for r in rows}
    def pct(value):return '—' if value is None else f'{value:.2%}'
    def table(selected):
        headers=['任务','轨迹数','同/跨帧对数']+[f'{model} 同轨迹 / 跨轨迹' for model in selected]
        lines=['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']
        for task in tasks:
            r=lookup[task,selected[0]]
            values=[task,str(trajectories[task]),f"{r['within_total']} / {r['cross_total']}"]
            for model in selected:
                v=lookup[task,model];values.append(f"{pct(v['within_accuracy'])} / {pct(v['cross_accuracy'])}")
            lines.append('| '+' | '.join(values)+' |')
        return '\n'.join(lines)
    text=['# 新 equal-keyframes benchmark 逐任务结果','',
          '阈值 0.1；按任务文字聚合；得分为正确帧对数 / 本任务对应模式的帧对数。',
          '共 40 个任务、12 组模型。Uncap the red pen 只有 1 条轨迹，因此跨轨迹得分为 —。','']
    for model in models:
        text.extend([f'## {model}','',table([model]),''])
    (args.output/'per_task.md').write_text('\n'.join(text))
    selected=['icl_27b_step4000','official_4b_uniform8_interp32']
    comparison=table(selected)
    (args.output/'best_icl_vs_official.md').write_text('# ICL 27B step 4000 与官方 4B 8 帧插值\n\n'+comparison+'\n')
    print(comparison)
    print(f'Validated {len(rows)} task/model rows against overall score numerators and denominators.')


if __name__=='__main__':main()
