"""Train/reload the multimodal FUSION task policy on all offline672 episodes.

This is a policy baseline, not the separately trained probabilistic world model
or REF. Reuses the existing architecture, CEWM losses and CPA replay code.
"""
import argparse
from collections import deque
import hashlib
import itertools
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from baseline672.data import load_episodes
from fusion.agent.fusion.losses import (compute_action_loss, compute_cewm_losses,
                                        ensure_finite_losses, mix_action_losses)
from fusion.agent.fusion.model import Fusion_Structure
from fusion.agent.fusion.risk_error_replay import SeverityCpaEventWindowDataset
from fusion.agent.fusion.rollout import predict_action_and_values
from fusion.agent.fusion.utils import _sample_window_start, discount_cumsum


FORMAT = "offline672-fusion-policy-v1"
NET = dict(state_dim=35, act_dim=2, n_blocks=3, h_dim=64, context_len=20,
           n_heads=1, drop_p=0.1, max_timestep=1000, use_bev=True)
RTG_SCALE, CTG_SCALE = 300., 10.


class Windows(torch.utils.data.Dataset):
    """One row per trajectory. Keep large BEVs as disk-backed views."""
    context_len = NET["context_len"]

    def __init__(self, episodes):
        self.trajectories = []
        for ep in episodes:
            n = len(ep["actions"])
            if not 0 < n <= NET["max_timestep"]:
                raise ValueError("episode length outside positional embedding")
            self.trajectories.append(dict(
                # The last observation has no executed action; don't pair it
                # with a synthetic action or another episode's first action.
                state=ep["states"][:-1], lidar=ep["lidars"][:-1], bev=ep["bevs"][:-1],
                actions=ep["actions"],
                rtg=discount_cumsum(ep["rewards"][:, None], 1.) / RTG_SCALE,
                ctg=discount_cumsum(ep["costs"][:, None], 1.) / CTG_SCALE))

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, index):
        row = self.trajectories[index]
        start = _sample_window_start(len(row["actions"]), self.context_len)
        return self.window_at(index, start)

    def window_at(self, index, start):
        row = self.trajectories[index]
        n = len(row["actions"])
        if not 0 <= start < n:
            raise IndexError("window starts outside the episode")
        count = min(self.context_len, n - start)
        def window(field):
            arr = row[field]
            # Make a writable tensor, without copying the entire mmap.
            result = np.zeros((self.context_len,) + arr.shape[1:], dtype=arr.dtype)
            result[:count] = arr[start:start + count]
            return torch.from_numpy(result)
        time = torch.zeros(self.context_len, dtype=torch.long)
        time[:count] = torch.arange(start, start + count)
        mask = torch.arange(self.context_len) < count
        return (time, (window("state"), window("lidar"), window("bev")),
                window("actions"), window("rtg"), window("ctg"), mask)


def on_device(batch, device):
    t, obs, a, r, c, mask = batch
    return (t.to(device), [x.to(device) for x in obs], a.to(device), r.to(device),
            c.to(device), mask.to(device))


def bisimulation_update(model, optimizer, batch):
    """Original separate tokenizer update, once per 500 ordinary updates."""
    t, obs, a, r, c, mask = batch
    p = torch.randperm(len(t), device=t.device)
    s1, _, r1, c1 = model(t, obs, a, r, c)
    with torch.no_grad():
        s2, _, r2, c2 = model(t[p], [x[p] for x in obs], a[p], r[p], c[p])
        target = (F.mse_loss(obs[0], obs[0][p]) + F.huber_loss(r, r[p])
                  + F.huber_loss(c, c[p]))
    b = mask.bool()
    pred = (F.mse_loss(s1[b], s2[b[p]]) + F.huber_loss(r1[b], r2[b[p]])
            + F.huber_loss(c1[b], c2[b[p]]))
    loss = F.mse_loss(pred, target)
    ensure_finite_losses({"bisim": loss})
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def repeat_loader(loader):
    while True:
        yield from loader


def train(data, output, steps=4000, batch_size=128, device="cpu", seed=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device)
    episodes = load_episodes(data)  # all 672, including the 115 failures
    ds = Windows(episodes)
    events = SeverityCpaEventWindowDataset(ds, score_threshold=.250)
    event_batch = min(len(events), max(1, min(32, batch_size // 4)))
    normal = repeat_loader(DataLoader(ds, batch_size=batch_size,
        sampler=RandomSampler(ds, replacement=True, num_samples=batch_size * 500)))
    special = repeat_loader(DataLoader(events, batch_size=event_batch, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 104729)))
    model = Fusion_Structure(**NET).to(device).train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    bis_params = itertools.chain.from_iterable(getattr(model, name).parameters()
        for name in ("embed_rtg", "embed_ctg", "embed_state", "embed_action",
                     "embed_lidar", "embed_bev", "embed_agg"))
    bis_optim = torch.optim.AdamW(bis_params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim,
        lambda k: 1. if k <= 25000 else min(.1, np.exp(-1e-4 * (k - 25000))))
    print("FUSION: {} episodes, {} transitions, {} critical windows; {} steps, device={}"
          .format(len(ds), sum(len(ep["actions"]) for ep in episodes),
                  len(events), steps, device), flush=True)
    for i in range(1, steps + 1):
        t, obs, a, r, c, m = on_device(next(normal), device)
        preds = model(t, obs, a, r, c)
        losses = compute_cewm_losses(*preds, obs[0], a, r, c, m)
        te, oe, ae, re, ce, me = on_device(next(special), device)
        epred = model(te, oe, ae, re, ce)[1]
        losses["action"] = mix_action_losses(losses["action"],
                                             compute_action_loss(epred, ae, me), .10)
        ensure_finite_losses(losses)
        total = sum(losses.values())
        optim.zero_grad()
        total.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), .25)
        if not torch.isfinite(grad):
            raise FloatingPointError("non-finite policy gradient")
        optim.step()
        scheduler.step()
        if i % 500 == 0:
            bisimulation_update(model, bis_optim, on_device(next(normal), device))
        if i == 1 or i % 100 == 0 or i == steps:
            print("update={} total={:.6f} action={:.6f}".format(
                i, total.item(), losses["action"].item()), flush=True)

    payload = dict(format=FORMAT, model=model.state_dict(), spec=NET, steps=steps, seed=seed,
                   batch_size=batch_size, event_batch=event_batch, critical_windows=len(events),
                   dataset="offline672-v1", manifest_sha256=hashlib.sha256(
                       (Path(data) / "manifest.json").read_bytes()).hexdigest())
    # Do not overwrite an earlier run, even if a concurrent trainer won the race.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(output.parent)) as tmp:
        torch.save(payload, tmp)
        tmp.flush()
        os.link(tmp.name, str(output))
    return payload


class Policy:
    """Causal rollout. Left-pad cold history exactly as the frozen task policy."""
    def __init__(self, model, device):
        self.model = model.eval()
        self.device = torch.device(device)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=NET["context_len"])
        self.step = 0
        self.rtg, self.ctg = 200. / RTG_SCALE, 0.
        self.awaiting = False

    @torch.no_grad()
    def act(self, obs):
        if self.awaiting or self.step >= NET["max_timestep"]:
            raise RuntimeError("call observe after act, and reset at episode boundaries")
        record = []
        for k, shape in (("state", (35,)), ("lidar", (240,)), ("bev", (5, 84, 84))):
            value = np.asarray(obs[k])
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("invalid FUSION observation: " + k)
            if k == "bev":
                if value.dtype == np.uint8:
                    value = value.astype(np.float32) / 255.
                elif value.min() < -1e-6 or value.max() > 1 + 1e-6:
                    raise ValueError("float BEV must be normalized in [0,1]")
            record.append(np.array(value, dtype=np.float32, copy=True))
        self.history.append(record + [np.zeros(2, np.float32), [self.rtg], [self.ctg], self.step])
        def history(j):
            seq = np.stack([entry[j] for entry in self.history])
            arr = np.zeros((NET["context_len"],) + seq.shape[1:],
                           dtype=np.int64 if j == 6 else np.float32)
            arr[-len(seq):] = seq
            return torch.from_numpy(arr).unsqueeze(0).to(self.device)
        s, l, b, a, r, c, t = [history(j) for j in range(7)]
        action = predict_action_and_values(self.model, t, s, l, a, r, c, bev=b,
                                           predict_values=False)[0][0].cpu().numpy()
        if not np.isfinite(action).all():
            raise FloatingPointError("FUSION returned non-finite action")
        self.awaiting = True
        return action

    def observe(self, action, reward, cost):
        if not self.awaiting:
            raise RuntimeError("act must precede observe")
        action = np.asarray(action, np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)) or not np.isfinite([reward, cost]).all():
            raise ValueError("invalid executed transition")
        self.history[-1][3] = action.copy()
        self.rtg = max(0., self.rtg - float(reward) / RTG_SCALE)
        self.ctg = max(0., self.ctg - float(cost) / CTG_SCALE)
        self.step += 1
        self.awaiting = False


def load_policy(checkpoint, device="cpu"):
    saved = torch.load(str(checkpoint), map_location="cpu")
    if saved.get("format") != FORMAT or saved.get("spec") != NET:
        raise ValueError("not an offline672 FUSION checkpoint")
    model = Fusion_Structure(**NET)
    model.load_state_dict(saved["model"], strict=True)
    return Policy(model.to(device), device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset/offline672-v1")
    p.add_argument("--output", required=True, help="new checkpoint file; never overwritten")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=2)
    opts = p.parse_args()
    torch.set_num_threads(opts.threads)
    train(opts.data, opts.output, opts.steps, opts.batch_size, opts.device, opts.seed)


if __name__ == "__main__":
    main()
