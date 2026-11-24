"""Train the offline672 adaptation of the repository CPQ; no sensor or env at train time."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baseline672.data import VERSION, load_episodes, sha256
from .cpq_core import CPQ, CPQMetaDriveConfig, SquashedGaussianMLPActor


SCHEMA = "offline672-cpq-policy-v1"


class Replay:
    """Uniform transitions, not uniform episodes. Never join s' across a reset."""
    def __init__(self, episodes):
        pieces = [[] for _ in range(6)]
        self.truncations = 0
        for ep in episodes:
            s, a = ep["states"], ep["actions"]
            n = len(a)
            r, c, term, trunc = [np.asarray(ep[k]) for k in
                                 ("rewards", "costs", "terminated", "truncated")]
            if (n < 1 or s.shape != (n+1, 35) or a.shape != (n, 2)
                    or any(v.shape != (n,) for v in (r, c, term, trunc))
                    or np.any(np.abs(a) > 1.000001)
                    or not all(np.isfinite(v).all() for v in (s, a, r, c))
                    or not np.isin(c, [0, 1]).all()
                    or not np.isin(term, [0, 1]).all() or not np.isin(trunc, [0, 1]).all()
                    or np.any(term[:-1]) or np.any(trunc[:-1])
                    or int(term[-1]) + int(trunc[-1]) != 1):
                raise ValueError("invalid CPQ episode/transition alignment")
            self.truncations += int(trunc[-1])
            # A timeout has a real final observation for bootstrapping, not a
            # crash / successful terminal / reset observation.
            for parts, v in zip(pieces, (s[:-1], s[1:], a, r, c, term)):
                parts.append(v)
        self.arrays = tuple(np.concatenate(parts).astype(np.float32, copy=False) for parts in pieces)

    def sample(self, rng, size, device):
        ix = rng.integers(len(self.arrays[0]), size=size)
        return tuple(torch.from_numpy(v[ix]).to(device) for v in self.arrays)


def cost_target(cost, terminated, next_q, gamma):
    if cost.ndim != 1 or terminated.shape != cost.shape or next_q.shape != cost.shape:
        raise ValueError("scalar targets must be [B], not [B,1] broadcasting to [B,B]")
    return cost + gamma * (1 - terminated) * next_q


class Offline672CPQ(CPQ):
    """Unchanged repository architecture/reward backup/actor/VAE; see method notes."""
    @torch.no_grad()
    def candidate_set(self, obs):
        # The distribution exposed by SquashedGaussianMLPActor is PRE tanh.
        _, _, dist = self.actor(obs, False, False, True)
        actions = self.max_action * dist.sample((self.sample_action_num,)).tanh()
        actions = actions.reshape(-1, self.action_dim)
        states = obs.repeat(self.sample_action_num, 1)
        _, mean, std = self.vae(states, actions)
        kl = -0.5 * (1 + 2 * std.log() - mean.square() - std.square()).mean(-1)
        selected = kl >= torch.quantile(kl, 0.75)
        return states, actions, selected

    def ood_cost(self, states, actions, mask):
        # Preserve repository reduction (unselected mass contributes 0), but
        # evaluate *online* Qc WITH gradients instead of a detached target Qc.
        candidate_q, _ = self.cost_critic.predict(states, actions)
        return (mask.to(candidate_q.dtype) * candidate_q).mean()

    def cost_critic_loss(self, observations, next_observations, actions, costs, done):
        with torch.no_grad():
            next_action, _ = self._actor_forward(next_observations, False, False)
            next_q, _ = self.cost_critic_old.predict(next_observations, next_action)
            target = cost_target(costs, done, next_q, self.gamma)
        _, qs = self.cost_critic.predict(observations, actions)
        qc_ood = self.ood_cost(*self.candidate_set(observations))
        mse = self.cost_critic.loss(target, qs)
        loss = mse - self.log_alpha.exp() * (qc_ood - self.qc_thres)
        self.cost_critic_optim.zero_grad()
        loss.backward()
        self.cost_critic_optim.step()
        with torch.no_grad():
            self.log_alpha.add_(self.alpha_lr * self.log_alpha.exp() *
                                (self.qc_thres - qc_ood.detach())).clamp_(-5., 5.)
        return loss, {"loss/cost_critic_loss": loss.item(), "loss/cost_td": mse.item(),
                      "ood_qc": qc_ood.item(), "alpha": self.log_alpha.exp().item()}


class DeploymentPolicy:
    def __init__(self, actor, device):
        self.actor, self.device = actor.eval(), device

    def reset(self):
        pass

    def observe(self, action, reward, cost):
        pass

    @torch.no_grad()
    def act(self, obs):
        s = np.asarray(obs["state"], dtype=np.float32)
        if s.shape != (35,) or not np.isfinite(s).all():
            raise ValueError("CPQ requires a finite current state35")
        a, _ = self.actor(torch.tensor(s, device=self.device)[None, :],
                          deterministic=True, with_logprob=False)
        return a[0].cpu().numpy()


def load_policy(checkpoint, device="cpu"):
    # PyTorch 1.10 checkpoints: only load your own or otherwise trusted files.
    payload = torch.load(str(checkpoint), map_location=device)
    if payload.get("schema") != SCHEMA or payload.get("algorithm") != "cpq":
        raise ValueError("not an offline672 CPQ policy checkpoint")
    actor = SquashedGaussianMLPActor(35, 2, payload["config"]["a_hidden_sizes"], nn.ReLU).to(device)
    actor.load_state_dict(payload["actor_state"], strict=True)
    return DeploymentPolicy(actor, device)


def save_policy(path, model, config, steps, summary):
    payload = dict(schema=SCHEMA, algorithm="cpq", dataset=VERSION, config=config,
                   steps=steps, data_summary=summary,
                   actor_state={k: v.detach().cpu() for k, v in model.actor.state_dict().items()})
    with Path(path).open("xb") as f:  # inference checkpoint, never overwrite a previous run
        torch.save(payload, f)


def train(args):
    path = Path(args.output)
    metrics = path.with_suffix(".metrics.csv")
    if path.exists() or metrics.exists() or path == metrics:
        raise FileExistsError("choose a fresh checkpoint and metrics filename")
    if min(args.steps, args.batch_size, args.threads) < 1 or not math.isfinite(args.cost_limit) or args.cost_limit < 0:
        raise ValueError("invalid update budget, batch size, threads or cost limit")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    episodes = load_episodes(args.data)
    replay = Replay(episodes)
    summary = dict(episodes=len(episodes), transitions=len(replay.arrays[0]),
                   costs=int(replay.arrays[4].sum()), truncations=replay.truncations,
                   dataset_manifest_sha256=sha256(Path(args.data)/"manifest.json"))
    if tuple(summary[k] for k in ("episodes", "transitions", "costs", "truncations")) != (672, 141345, 115, 1):
        raise ValueError("unexpected offline672 CPQ training coverage")
    defaults = CPQMetaDriveConfig()
    arch = {k: getattr(defaults, k) for k in
            ("a_hidden_sizes", "c_hidden_sizes", "vae_hidden_sizes", "sample_action_num",
             "gamma", "tau", "beta", "num_q", "num_qc", "qc_scalar", "episode_len")}
    arch.update(state_dim=35, action_dim=2, max_action=1., cost_limit=args.cost_limit)
    model = Offline672CPQ(**arch, device=args.device)
    lr = {k: getattr(defaults, k) for k in ("actor_lr", "critic_lr", "alpha_lr", "vae_lr")}
    model.setup_optimizers(**lr)
    config = dict(arch, **lr, seed=args.seed, batch_size=args.batch_size, reward_scale=1., cost_scale=1.,
                  threads=args.threads, device=args.device, training_steps=args.steps)
    print(json.dumps({"data": summary, "config": config}, sort_keys=True), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("x", encoding="utf-8", newline="") as f:
        writer = None
        for step in range(1, args.steps+1):
            s, ns, a, r, c, terminal = replay.sample(rng, args.batch_size, args.device)
            stats = {}
            for _, vals in (model.vae_loss(s, a), model.critic_loss(s, ns, a, r, terminal),
                            model.cost_critic_loss(s, ns, a, c, terminal), model.actor_loss(s)):
                stats.update(vals)
            model.sync_weight()
            if not all(math.isfinite(v) for v in stats.values()):
                raise FloatingPointError("nonfinite CPQ training stats at step %d" % step)
            if step == 1 or step % 100 == 0 or step == args.steps:
                row = dict(step=step, **stats)
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()
            if step == 1 or step % 1000 == 0 or step == args.steps:
                print("step=%d cost_td=%.6g actor_loss=%.6g" %
                      (step, stats["loss/cost_td"], stats["loss/actor_loss"]), flush=True)
    save_policy(path, model, config, args.steps, summary)
    print("saved CPQ deployment checkpoint: %s" % path, flush=True)


def main():
    defaults = CPQMetaDriveConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=defaults.update_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cost-limit", type=float, default=float(defaults.cost_limit))
    train(parser.parse_args())


if __name__ == "__main__":
    main()
