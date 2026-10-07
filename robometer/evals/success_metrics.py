"""Final-outcome classification on whole trajectories, with explicit coverage."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def final_success_metrics(results):
    labels, probabilities = [], []
    seen = set()
    for row in results:
        quality = str(row.get('quality_label', '')).lower()
        if quality not in {'successful', 'failure', 'failed', 'suboptimal'} or row.get('success_probs') is None:
            continue
        identity = (row.get('data_source'), row.get('id'), row.get('task'))
        if row.get('id') is not None and identity in seen:
            raise ValueError('Final success evaluation requires one whole-trajectory prediction per ID')
        seen.add(identity)
        labels.append(int(quality == 'successful'))
        probabilities.append(float(np.asarray(row['success_probs']).reshape(-1)[-1]))
    y, p = np.asarray(labels), np.asarray(probabilities)
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite success probabilities')
    out = {'success_n': len(y), 'success_positive_n': int(y.sum()),
           'success_negative_n': int(len(y) - y.sum()), 'success_skipped_n': len(results) - len(y)}
    if len(y):
        pred = p > 0.5
        out['success_accuracy_0.5'] = float(np.mean(pred == y))
        out['success_brier'] = float(np.mean((p-y)**2))
        if (y == 1).any():
            out['success_positive_acc_0.5'] = float(np.mean(pred[y == 1]))
        if (y == 0).any():
            out['success_negative_acc_0.5'] = float(np.mean(~pred[y == 0]))
        if len(np.unique(y)) == 2:
            out['success_auprc'] = float(average_precision_score(y, p))
            out['success_auroc'] = float(roc_auc_score(y, p))
    return out
