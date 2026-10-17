"""Audit paired samples and summarize original and extended model comparisons."""
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
    parser.add_argument('--models', help='Comma-separated names; defaults to original models plus models.json')
    args = parser.parse_args()
    root = args.output.resolve()
    names = ['official4b'] + [f'qwen27b_step{s}' for s in [1000, 2000, 3000, 4000, 5000]]
    if (root / 'models.json').exists():
        names += [job['name'] for job in json.loads((root / 'models.json').read_text())]
    if args.models:
        names = args.models.split(',')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate model names')
    reference, audits, overview, details, order_details = None, {}, [], [], []
    for name in names:
        path = root / name
        if not (path / 'COMPLETE.json').exists():
            raise RuntimeError(f'Not completed: {name}')
        metrics = json.loads((path / 'all_metrics.json').read_text())
        audit = {}
        order_rows = []
        for result_path in sorted(path.glob('*/*_results.json')):
            rows = json.loads(result_path.read_text())
            for row in rows:
                for key in ['progress_pred', 'success_probs', 'prediction_prob', 'preference_pred', 'target_progress']:
                    if key in row and not finite_tree(row[key]):
                        raise ValueError(f'Nonfinite values: {result_path}: {key}')
            identities = [json.dumps(identity(row), sort_keys=True) for row in rows]
            digest = hashlib.sha256('\n'.join(sorted(identities)).encode()).hexdigest()
            audit[str(result_path.relative_to(path))] = {'n': len(rows), 'sha256': digest}
            if result_path.parent.name == 'quality_preference':
                by_label = {label: [r for r in rows if r['preference_labels'] == label] for label in [0, 1]}
                if not by_label[0] or len(by_label[0]) != len(by_label[1]):
                    raise ValueError(f'Preference directions are not balanced: {result_path}')
                directional = {'model': name, 'dataset': result_path.name.removesuffix('_results.json'),
                               'n_pairs': len(by_label[0])}
                for label, column in [(1, 'accuracy_A_preferred'), (0, 'accuracy_B_preferred')]:
                    directional[column] = statistics.mean(float(r['preference_pred'] == label) for r in by_label[label])
                directional['A_minus_B_accuracy'] = directional['accuracy_A_preferred'] - directional['accuracy_B_preferred']
                order_rows.append(directional)
        if len(audit) != 24:
            raise ValueError(f'Expected 4 eval types x 6 datasets, got {len(audit)}: {name}')
        if reference is None:
            reference = audit
        if audit != reference:
            raise ValueError(f'Evaluation samples differ for {name}')
        audits[name] = audit
        summary = {'model': name}
        load_report = json.loads((path / 'checkpoint_load_report.json').read_text())
        state_path = Path(load_report['checkpoint']) / 'trainer_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        summary.update(training_step=state.get('global_step'), training_epoch=state.get('epoch'))
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
            if len(values) != 6:
                raise ValueError(f'Expected six finite dataset metrics: {name}/{column}: {len(values)}')
        summary['seconds'] = json.loads((path / 'COMPLETE.json').read_text())['seconds']
        for column in ['accuracy_A_preferred', 'accuracy_B_preferred', 'A_minus_B_accuracy']:
            summary['preference_' + column] = statistics.mean(r[column] for r in order_rows)
        if not math.isclose((summary['preference_accuracy_A_preferred'] + summary['preference_accuracy_B_preferred']) / 2,
                            summary['preference_accuracy_balanced'], abs_tol=1e-12):
            raise ValueError(f'Directional preference scores disagree with balanced accuracy: {name}')
        overview.append(summary)
        details.extend(per_dataset.values())
        order_details.extend(order_rows)
    (root / 'sample_audit.json').write_text(json.dumps({'all_models_identical_samples': True, 'models': audits}, indent=2))
    (root / 'comparison.json').write_text(json.dumps(overview, indent=2))
    for name, rows in [('comparison.csv', overview), ('per_dataset.csv', details),
                       ('preference_order.csv', order_details)]:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (root / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extended = len(names) != 6 or names[-1] != 'qwen27b_step5000'
    fig, axes = plt.subplots(2, 3, figsize=(18, 11) if extended else (14, 7), constrained_layout=True)
    keys = ['ranking_accuracy_last', 'ranking_kendall_last', 'preference_accuracy_balanced',
            'progress_mse', 'progress_pearson', 'success_auprc']
    titles = ['Trajectory ranking accuracy (higher better)', 'Ranking Kendall tau (higher better)',
              'Pairwise preference accuracy (higher better)', 'Progress MSE (lower better)',
              'Progress Pearson r (higher better)', 'Final success AP (higher better)']
    for ax, key, title in zip(axes.flat, keys, titles):
        if extended:
            labels = [r['model'].replace('_step', ' / ').replace('trained4b', 'Trained 4B')
                      .replace('qwen27b', '27B').replace('qwen397b', '397B')
                      .replace('official4b', 'Official 4B') for r in overview]
            colors = ['tab:orange' if r['model'] == 'official4b' else
                      'tab:green' if r['model'].startswith('trained4b') else
                      'tab:red' if r['model'].startswith('qwen397b') else 'tab:blue' for r in overview]
            ax.barh(labels, [r[key] for r in overview], color=colors)
            ax.invert_yaxis()
            ax.tick_params(axis='y', labelsize=9)
        else:
            ax.plot([1000, 2000, 3000, 4000, 5000], [r[key] for r in overview[1:]], marker='o', label='Qwen 27B')
            ax.axhline(overview[0][key], color='tab:orange', linestyle='--', label='Official Robometer 4B')
            ax.set_xlabel('Training step')
            ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=.2)
    fig.suptitle('Same OOD samples; unweighted mean across datasets')
    fig.savefig(root / 'comparison.png', dpi=170)
    print(json.dumps(overview, indent=2))


if __name__ == '__main__':
    main()
