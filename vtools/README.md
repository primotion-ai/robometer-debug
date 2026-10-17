# Robometer 数据集浏览器

使用项目现有 uv 虚拟环境，无需安装新依赖。只读访问解包后的数据。

```bash
cd /mnt/workspace/liuzihao/project/robometer
uv run --no-sync python vtools/app.py --host 127.0.0.1 --port 7860
```

浏览器打开 `http://127.0.0.1:7860`（远程机器需用服务器地址，或 SSH 转发端口）。
默认绑定地址为 `127.0.0.1`；`0.0.0.0` 用于可信网络，无登录认证。

从另一台电脑访问服务器上的服务，可在那台电脑执行：

```bash
ssh -N -L 7860:127.0.0.1:7860 liuzihao@服务器地址
```

然后在电脑上打开 `http://127.0.0.1:7860`。

当前后台会话及日志：

```bash
tmux -L robometer-vtools attach -t robometer-vtools
tail -f /tmp/robometer-vtools.log
# 停止网页服务
tmux -L robometer-vtools kill-session -t robometer-vtools
```

默认数据目录为 `/mnt/workspace/dataset/robometer-1m`，可通过
`--data-root /path/to/data` 或 `ROBOMETER_PROCESSED_DATASETS_PATH` 指定。

- 子集名称筛选、轨迹 ID 搜索、每页 100 条记录。
- 逐帧预览、拖动进度条、播放/暂停。预览速度不是原始视频 FPS。
- 全部 Arrow 字段（包括语言向量）、质量标签、索引分组、子集信息和字段定义。
- 按需读取特征文件中的完整张量数值，支持下载原始 NPZ/PT。
- 兼容旧的 `processed_datasets/` 前缀和原始机器路径；从子集的 `frames/`、
  `embeddings/` 目录定位文件。图像读取 `frames` 字段，不使用原始视频 `frames_path`。
- Arrow 使用内存映射，最多缓存两个子集表和两条轨迹的帧数组。

这里只展示缓存里实际保存的数据，不生成奖励、动作或原始视频时间戳。

本目录的 `outputs*/`、`runs/`、缓存、日志及本地私密配置已被 Git 忽略，仓库仅保存浏览器代码和说明。
