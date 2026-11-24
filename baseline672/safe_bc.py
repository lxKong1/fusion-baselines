"""State-only safe behavioral cloning on verified safe successful episodes."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fusion.agent.bc.encoder import StateEncoder


SCHEMA = "baseline672.safe-bc-state.v1"


class SafeBC(nn.Module):
    def __init__(self):
        super().__init__()
        # Share the original BC state branch (35 -> 64 -> 64 -> 2*2).
        self.state_net = StateEncoder(hidden_dim=64, output_dim=2)

    def parameters_at(self, states):
        mean, log_std = self.state_net.encoder(states).chunk(2, dim=-1)
        return mean.tanh(), log_std.clamp(-5, 2)

    def loss(self, states, actions):
        mean, log_std = self.parameters_at(states)
        return (0.5 * ((actions - mean) * (-log_std).exp()).square() + log_std).mean()

    def reset(self):
        pass

    @torch.no_grad()
    def act(self, obs):
        x = torch.as_tensor(obs["state"], dtype=torch.float32, device=next(self.parameters()).device)
        if x.shape != (35,):
            raise ValueError("expected state35")
        return self.parameters_at(x)[0].cpu().numpy()

    def observe(self, action, reward, cost):
        pass


def pairs(episodes):
    states, actions = [], []
    for ep in episodes:
        if ep["outcome"] != "success" or np.any(ep["costs"]) or np.any(ep["truncated"]):
            raise ValueError("unsafe or incomplete episode in Safe BC view")
        x, a = ep["states"][:-1], ep["actions"]
        if x.shape != (len(a), 35) or a.shape != (len(a), 2):
            raise ValueError("misaligned observations and actions")
        states.append(x)
        actions.append(a)
    return tuple(torch.from_numpy(np.concatenate(rows).astype(np.float32, copy=False))
                 for rows in (states, actions))


def load_policy(checkpoint, device="cpu"):
    record = torch.load(checkpoint, map_location=device)
    if record["schema"] != SCHEMA:
        raise ValueError("not a Safe BC v1 checkpoint")
    net = SafeBC().to(device)
    net.load_state_dict(record["weights"])
    return net.eval()


def main():
    from baseline672.data import load_episodes
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="dataset/offline672-v1")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if min(args.steps, args.batch_size) < 1:
        p.error("steps and batch size must be positive")
    if args.output.exists():
        p.error("refusing to overwrite an existing checkpoint")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    episodes = load_episodes(args.data, safe_only=True)
    if len(episodes) != 556:
        raise ValueError("offline672-v1 must have 556 successful cost-free episodes")
    x, a = pairs(episodes)
    net = SafeBC().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    print(f"safe_bc: {len(episodes)}/672 episodes, {len(a)} real actions", flush=True)
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(a), (args.batch_size,))
        loss = net.loss(x[idx].to(args.device), a[idx].to(args.device))
        if not torch.isfinite(loss):
            raise ValueError("non-finite BC loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 1000 == 0 or step == args.steps:
            print(f"step={step} loss={loss.item():.5f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as f:
        torch.save(dict(schema=SCHEMA, weights=net.state_dict(), dataset="offline672-v1",
                        seed=args.seed, steps=step, batch_size=args.batch_size), f)


if __name__ == "__main__":
    main()
