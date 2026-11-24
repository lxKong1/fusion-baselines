# CPQ：offline672-v1 适配说明

这是 **CPQ 的本仓库适配版**，而不是原论文结果的复现声明。
复用 上游 FUSION CPQ 的网络、VAE、奖励 Critic、Actor 和软更新逻辑，
算法入口是 `python -m baseline672.cpq_policy`。当前配置和所需网络都在 `baseline672/cpq_core.py`；以下是此适配版的差异。

## 用法

在仓库根目录、已激活复现环境并解压数据后：

```bash
# 仅打通：全量数据、8次随机 minibatch 更新；不是性能结果
python -m baseline672.cpq_policy --data dataset/offline672-v1 --output output/cpq-smoke.pt --steps 8 --batch-size 128 --seed 1 --device cpu
# 正式预算示例：新文件名。也可使用经验证可用的 --device cuda:0
python -m baseline672.cpq_policy --data dataset/offline672-v1 --output output/cpq-seed1.pt --steps 100000 --batch-size 512 --seed 1 --device cpu
```

同名权重和 `.metrics.csv` 拒绝覆盖。日志记录训练更新（100次一次，包括首末更新）；
权重内保存 actor、动作/网络规格、种子、预算和数据清单 SHA256。它是**部署权重**，
不可用来恢复中断的优化器。训练不启动 W&B / MetaDrive，不需要账号。
部署只读取当前 `obs["state"]`（35维），确定性输出两维动作，不用 LiDAR、BEV、REF 或其他旧权重；
请只加载来源可信的 PyTorch 检查点。

## 数据与更新口径

- 全部 672 条的 **141,345 个实际执行转移均匀抽样**；115 个碰撞/离路即时 cost 保持原始 0/1。
  同一条轨迹内配对 `s[t],a[t],r[t],c[t],s[t+1]`，不跨回合拼接。
- 奖励和代价均使用原始单位，倍数都是 **1**。旧配置声明过 `reward_scale=0.1`，
  但旧 `CPQTrainer.train_one_step` 实际直接使用了原始张量，本版把实际行为显式保存。
  奖励和代价的一步目标均在真实终态停止 bootstrap；超时使用最终真实观测继续 bootstrap，
  不是把 reset 观测接进上一条轨迹。标量张量形状固定为 `[batch]`，避免把 Bellman 误差广播成 batch×batch。
- 保留 VAE KL 前 25% 候选作为分布外样本的筛选方案（批次级分位点，含相等项），
  保留按全部候选数而非仅按命中数归一的惩罚约定。
  旧实现的 `qc_ood` 在 `no_grad` 里由目标网络计算，进入 loss 却不向在线代价 Critic 传梯度；
  新版只停止动作、筛选和目标的梯度，选中动作的 Qc 由**在线代价 Critic**重新计算。
  且将采自高斯的候选经 `tanh` 转换为训练和环境一致的合法动作。
- 默认沿用配置 `gamma=0.99`、`tau=0.005`、`episode_len=1000`、`cost_limit=10`；
  可显式设置 `--cost-limit`。其在 Critic 门控中是
  `q_thres=cost_limit*(1-gamma**episode_len)/((1-gamma)*episode_len)`，
  并非“每个动作的事故概率阈值”。实际数据每回合至多一次失败，在该标度下默认预算偏宽松，
  为保留可追溯的起点而非已调优的安全结果。若调预算，应单独记下并使用独立评估种子。
- 冒烟测试只验优化、梯度、保存和加载；是否比 Safe BC 等更安全，需要正式训练与闭环测试。

```bash
python -m pytest -q tests/test_baseline672_cpq.py
```
