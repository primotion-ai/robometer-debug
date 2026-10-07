"""Audit paired samples and summarize a completed six-model benchmark."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

METRICS = {
    'progress_mse': ('reward_alignment', 'loss'),
    'progress_pearson': ('reward_alignment', 'pearson'),
    'ranking_kendall_last': ('policy_ranking', 'kendall_last'),
    'ranking_accuracy_last': ('policy_ranking', 'ranking_acc_last'),
    'preference_accuracy_balanced': ('quality_preference', 'preference_accuracy'),
    'success_auprc': ('policy_ranking', 'success_auprc'),
    'success_auroc': ('policy_ranking', 'success_auroc'),
    'success_accuracy_0.5': ('policy_ranking', 'success_accuracy_0.5'),
    'success_brier': ('policy_ranking', 'success_brier'),
    'task_confusion_margin': ('confusion_matrix', 'normalized_trace_minus_offdiag'),
}


def finite_tree(value):
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    return not isinstance(value, float) or math.isfinite(value)


def identity(row):
    keys = ['id', 'task', 'data_source', 'quality_label', 'partial_success',
            'target_progress', 'preference_labels', 'metadata', 'video_path']
    return {k: row[k] for k in keys if k in row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    names = ['official4b'] + [f'qwen27b_step{s}' for s in [1000, 2000, 3000, 4000, 5000]]
    reference, audits, overview, details = None, {}, [], []
    for name in names:
        path = root / name
        if not (path / 'COMPLETE.json').exists():
            raise RuntimeError(f'Not completed: {name}')
        metrics = json.loads((path / 'all_metrics.json').read_text())
        audit = {}
        for result_path in sorted(path.glob('*/*_results.json')):
            rows = json.loads(result_path.read_text())
            for row in rows:
                for key in ['progress_pred', 'success_probs', 'prediction_prob', 'preference_pred', 'target_progress']:
                    if key in row and not finite_tree(row[key]):
                        raise ValueError(f'Nonfinite values: {result_path}: {key}')
            identities = [json.dumps(identity(row), sort_keys=True) for row in rows]
            digest = hashlib.sha256('\n'.join(sorted(identities)).encode()).hexdigest()
            audit[str(result_path.relative_to(path))] = {'n': len(rows), 'sha256': digest}
        if len(audit) != 24:
            raise ValueError(f'Expected 4 eval types x 6 datasets, got {len(audit)}: {name}')
        if reference is None:
            reference = audit
        if audit != reference:
            raise ValueError(f'Evaluation samples differ for {name}')
        audits[name] = audit
        summary = {'model': name}
        per_dataset = {}
        for column, (kind, suffix) in METRICS.items():
            values = []
            for key, value in metrics[kind].items():
                dataset, metric = key.rsplit('/', 1)
                if metric == suffix and isinstance(value, (int, float)) and math.isfinite(value):
                    values.append(value)
                    per_dataset.setdefault(dataset, {'model': name, 'dataset': dataset})[column] = value
            summary[column] = statistics.mean(values) if values else None
            summary[column + '_datasets'] = len(values)
        summary['seconds'] = json.loads((path / 'COMPLETE.json').read_text())['seconds']
        overview.append(summary)
        details.extend(per_dataset.values())
    (root / 'sample_audit.json').write_text(json.dumps({'all_models_identical_samples': True, 'models': audits}, indent=2))
    (root / 'comparison.json').write_text(json.dumps(overview, indent=2))
    for name, rows in [('comparison.csv', overview), ('per_dataset.csv', details)]:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (root / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    keys = ['ranking_accuracy_last', 'ranking_kendall_last', 'preference_accuracy_balanced',
            'progress_mse', 'progress_pearson', 'success_auprc']
    titles = ['Trajectory ranking accuracy (higher better)', 'Ranking Kendall tau (higher better)',
              'Pairwise preference accuracy (higher better)', 'Progress MSE (lower better)',
              'Progress Pearson r (higher better)', 'Final success AP (higher better)']
    for ax, key, title in zip(axes.flat, keys, titles):
        ax.plot([1000, 2000, 3000, 4000, 5000], [r[key] for r in overview[1:]], marker='o', label='Qwen 27B')
        ax.axhline(overview[0][key], color='tab:orange', linestyle='--', label='Official Robometer 4B')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Training step')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Same OOD samples; unweighted mean across datasets')
    fig.savefig(root / 'comparison.png', dpi=170)
    print(json.dumps(overview, indent=2))


if __name__ == '__main__':
    main()
