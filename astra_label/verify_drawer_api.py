"""Audit unchanged API scores and compare them with the saved local-login run."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE / 'outputs_api' / 'close_the_drawer'
LOCAL = PACKAGE / 'outputs' / 'close_the_drawer'
KEYFRAMES = [0, 17, 22, 25, 31]
REFERENCE = '1d035ec5-5127-4af9-9cca-fe214830d670'


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    assert len(summary) == 6
    assert json.loads((ROOT / 'errors.json').read_text()) == []
    shared = json.loads((ROOT / 'reference_calibration/rubric.json').read_text())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    comparisons = []
    anchor_checks = []
    for axis, item in zip(axes.flat, summary):
        ident = item['id']
        result = json.loads((ROOT / ident / 'labels.json').read_text())
        raw = json.loads((ROOT / ident / 'calls/target_000000_000031/response.json').read_text())
        baseline = json.loads((LOCAL / ident / 'labels.json').read_text())
        baseline_raw = json.loads((LOCAL / ident / 'calls/target_000000_000031/response.json').read_text())
        assert result['labels'] == raw['labels']
        assert baseline['labels'] == baseline_raw['labels']
        assert result['rubric'] == shared
        assert result['connection'] == {'auth_mode': 'api', 'base_url': 'https://ckffai.com/v1', 'proxy': 'direct'}
        assert [v['frame_index'] for v in result['labels']] == list(range(32))
        assert all(0 <= v['progress'] <= 1 for v in result['labels'])
        api_y = [v['progress'] for v in result['labels']]
        local_y = [v['progress'] for v in baseline['labels']]
        axis.plot(range(32), local_y, '--', color='gray', label='Local login (previous run)')
        axis.plot(range(32), api_y, color='tab:orange', marker='.', markersize=3, label='API (new run)')
        if ident == REFERENCE:
            for index, expected in zip(KEYFRAMES, [0, .25, .5, .75, 1]):
                actual = api_y[index]
                anchor_checks.append({'frame_index': index, 'expected': expected, 'actual': actual,
                                      'matches': actual == expected})
                axis.scatter(index, actual, marker='D', facecolors='white', edgecolors='black', zorder=5, clip_on=False)
                axis.annotate(f'idx {index}', (index, actual),
                              xytext=(-8, -20) if index == 31 else (6, 10), textcoords='offset points',
                              ha='right' if index == 31 else 'left', fontsize=8,
                              bbox={'facecolor': 'white', 'alpha': .8, 'edgecolor': 'none', 'pad': 1})
        axis.set(title=ident[:8] + (' (reference)' if ident == REFERENCE else ''),
                 xlabel='Stored frame index', ylabel='Progress', xlim=(0, 31), ylim=(-.03, 1.03))
        axis.grid(alpha=.2)
        comparisons.append({'id': ident, 'local_final': local_y[-1], 'api_final': api_y[-1],
                            'api_max': max(api_y), 'review_frames': item['needs_review_frames']})
    axes.flat[0].legend(fontsize=8)
    fig.suptitle('GPT-6 Astra high | direct model scores: previous local run vs new API run', fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT / 'comparison_with_local.png', dpi=180)
    plt.close(fig)
    with (ROOT / 'all_labels.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 192
    report = {'verified_frame_count': 192, 'scores_equal_raw_model_responses': True,
              'local_baseline_equal_original_responses': True, 'reference_anchors': anchor_checks,
              'comparison': comparisons}
    (ROOT / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Close the drawer：API 版标注', '',
             '本次通过 `https://ckffai.com/v1`，以 `gpt-6-astra`、`high` 调用 Codex Responses，关闭代理。'
             '结果独立保存在 `astra_label/outputs_api/close_the_drawer`；原本机结果仍在 `astra_label/outputs/close_the_drawer`。', '',
             '参考轨迹：`' + REFERENCE + '`。关键帧 `0、17、22、25、31`，参考分值 `0、0.25、0.5、0.75、1`。'
             '一次生成共用参考判据，随后六次分别标注每条轨迹的完整 32 帧。', '',
             '已核对 192 帧完整覆盖，JSON/CSV 分值与模型原始响应一致，无插值、平滑或分数校正。'
             '每条轨迹的 `calls/` 保留 prompt、schema、图片清单、原始响应和调用日志。', '',
             '![API 版曲线](comparison.png)', '',
             '## 与原本机结果比较', '',
             '![本机与 API 对比](comparison_with_local.png)', '',
             '| 轨迹 ID 前缀 | 本机末帧 | API 末帧 | API 待复核帧数 |',
             '| --- | --- | --- | --- |']
    for item in comparisons:
        ident = item['id']
        lines.append(f"| [{ident[:8]}]({ident}/labels.json) | {item['local_final']:.2f} | {item['api_final']:.2f} | {len(item['review_frames'])} |")
    lines += ['', '两次运行分别生成了参考阶段判据，且模型输出存在采样差异；该图是运行结果对照，不能单独据此归因于 API 或登录方式。', '',
              '## 参考关键帧核对', '', '| index | 预期 | API 原始输出 | 一致 |', '| --- | --- | --- | --- |']
    for row in anchor_checks:
        lines.append(f"| {row['frame_index']} | {row['expected']} | {row['actual']} | {'是' if row['matches'] else '否，需复核'} |")
    lines += ['', '文件：`all_labels.csv` 是 192 帧汇总；`summary.json` 是逐轨迹统计；`verification.json` 是校验和对照结果；'
              '`all_trajectories.png` 是分轨迹曲线。每条完整 ID 目录中有单独的 `labels.json`、`labels.csv`、`progress.png`。', '',
              '重新绘图：在项目根目录运行 `uv run --no-sync python astra_label/run_drawer_batch.py --auth-mode api --plot-only`。']
    (ROOT / 'README.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
