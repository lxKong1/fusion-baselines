# FUSION Baselines · MetaDrive 离线驾驶

用同一份 **672 条离线轨迹**从头训练 FUSION、Safe BC 或 CPQ，再把**自己训练出的策略**放入 MetaDrive 实际驾驶。代码与[训练数据](https://github.com/lxKong1/fusion-baselines/releases/tag/offline672-v1)在同一个仓库。

## 1. 获取代码和配置环境

以下命令在 Linux 上执行。需要 Git、Conda 和约 6 GiB 的数据空间（环境另计）。短训练和无窗口驾驶都支持 CPU。

```bash
git clone https://github.com/lxKong1/fusion-baselines.git
cd fusion-baselines
conda env create -f environment.yml
conda activate fusion-baselines-v1
bash scripts/install_metadrive.sh
python -m pip check
```

所有命令都在仓库根目录运行。如果环境已建好，直接激活即可。我们使用 Python 3.8.20、PyTorch 1.10.1、Gym 0.21 和 [FUSION 作者的 MetaDrive fork](https://github.com/HenryLHH/metadrive_clean)；安装脚本固定了提交并修正其 Gym 依赖声明，请勿替换为 PyPI 最新版。环境文件面向 Linux x86-64。

## 2. 下载数据

```bash
curl -fL --retry 3 -O https://github.com/lxKong1/fusion-baselines/releases/download/offline672-v1/offline672-v1.zip
python -m baseline672.data unpack --archive offline672-v1.zip
python -m baseline672.data verify
```

也可以浏览器进入上面的发布页下载 **offline672-v1.zip**（不是 Source code）。下载约 146 MiB，解包到 `dataset/offline672-v1/`，约 4.9 GiB。解包时校验文件，不覆盖已存在的目录。ZIP SHA256：`d7a9096d3b418628ef3a154159b4166c78de21458212a3f3a061681967d4dbf4`。

| 策略 | 使用的数据 | 观察输入 |
|---|---|---|
| FUSION | 全部 672 条 | 20 步历史，含 state35、LiDAR、BEV、动作及回报/代价条件 |
| Safe BC | 筛出的 556 条成功且全程零 cost 的轨迹（124,293 个动作） | 当前 state35 |
| CPQ | 全部 672 条，含 115 个正 cost 转移 | 当前 state35 |

数据来自 472 条人工驾驶、200 条 FUSION 驾驶，保留 141,345 个有效转移；556 次成功、77 次碰撞、38 次离路、1 次超时。没有预先拆分训练/验证/测试集。每条 T 步轨迹包含 T+1 帧观测（35 维状态、240 维 LiDAR、5×84×84 的语义鸟瞰图），以及 T 个二维动作、奖励、碰撞或离路代价、终止标记和超时标记。`manifest.json` 记录轨迹来源、场景、种子、结局及校验信息。

## 3. 选一组命令，先训练再开车

三组命令彼此独立。短训练只验证流程，不衡量性能。重复实验请换新的输出文件名，避免覆盖。

**FUSION**

```bash
python -m baseline672.fusion_policy --data dataset/offline672-v1 --output output/fusion-smoke.pt --steps 2 --batch-size 4 --seed 1 --device cpu
python -m baseline672.drive --algo fusion --checkpoint output/fusion-smoke.pt --scene O --traffic-density 0.2 --seed 940000
```

**Safe BC**

```bash
python -m baseline672.safe_bc --data dataset/offline672-v1 --output output/safe-bc-smoke.pt --steps 16 --seed 1 --device cpu
python -m baseline672.drive --algo safe-bc --checkpoint output/safe-bc-smoke.pt --scene O --traffic-density 0.2 --seed 940000
```

**CPQ**

```bash
python -m baseline672.cpq_policy --data dataset/offline672-v1 --output output/cpq-smoke.pt --steps 8 --batch-size 128 --seed 1 --device cpu
python -m baseline672.drive --algo cpq --checkpoint output/cpq-smoke.pt --scene O --traffic-density 0.2 --seed 940000
```

默认无窗口；本机有图形桌面时在驾驶命令后加 **`--render`** 看 MetaDrive 原生视图。可用 `--episodes 3` 连续驾驶三回合，`--traffic-density 0.3` 调流量，`--scene` 可选 S/O/R/T/X。车辆执行加载的策略动作（转向、油门/制动），没有外部接管；每回合打印步数、终止原因、累计奖励、代价和权重摘要。只加载可信的 PyTorch 权重。

较长训练可从以下预算起步，在上面的命令里替换 `--steps`、`--batch-size` 并更换 `--output` 路径：FUSION 为 4,000 次更新 / batch 128，Safe BC 为 50,000 / 128，CPQ 为 100,000 / 512。驾驶时 `--checkpoint` 指向新文件。它们是示例，不是统一或调优过的比较预算；正式比较请固定场景、密度及独立评估种子。

## 4. 适配和扩展

- [FUSION](baseline672/fusion_policy.py) 保留策略骨干、关键制动事件抽样和辅助损失；在本数据集上重新计算回报、代价与事件。部署初始目标为 RTG=200、CTG=0。
- [Safe BC](baseline672/safe_bc.py) 按轨迹真实结局和 cost 筛选，用原 StateEncoder 的 35→64→64→4 状态分支模仿动作，推理取有界均值。
- [CPQ](docs/cpq.md) 说明代价 Critic 的梯度、动作范围与超时处理。本仓库提供适配版，不声称复现论文指标。
- 添加下一个方法时，复用 [load_episodes](baseline672/data.py)，为自己的 checkpoint 实现 `reset()`、`act(obs)`、`observe(action,reward,cost)`，在[驾驶入口](baseline672/drive.py)的 `MODULES` 注册；如果新策略使用 BEV，还需启用相应环境观测。附训练和驾驶两条命令即可。
- 单元测试：`python -m pytest -q tests`（不需要下载完整数据）。

策略实现改编自 [FUSION](https://github.com/HenryLHH/fusion)，保留上游[软件许可证](LICENSE)及代码署名；MetaDrive 通过其上游单独安装。软件许可证不为另行发布的轨迹数据授予额外许可。
