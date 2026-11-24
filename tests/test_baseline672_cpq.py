import numpy as np
import pytest
import torch

from baseline672.cpq_policy import (
    DeploymentPolicy, Offline672CPQ, Replay, cost_target, load_policy, save_policy,
)


def small():
    torch.manual_seed(14)
    m = Offline672CPQ(35, 2, 1., [8], [8], vae_hidden_sizes=8,
                      sample_action_num=5, num_q=2, num_qc=2)
    m.setup_optimizers(1e-3, 1e-3, 1e-3, 1e-3)
    return m


def episode(base, terminal):
    return dict(states=np.repeat((np.arange(4)+base)[:, None], 35, axis=1),
                actions=np.zeros((3, 2)), rewards=np.array([0., 0., 2.]),
                costs=np.array([0., 0., float(terminal)]),
                terminated=np.array([False, False, terminal]),
                truncated=np.array([False, False, not terminal]))


def test_replay_keeps_episode_alignment_and_timeout_bootstrap():
    replay = Replay([episode(0, True), episode(40, False)])
    s, ns, a, r, c, d = replay.arrays
    assert s[:, 0].tolist() == [0, 1, 2, 40, 41, 42]
    assert ns[:, 0].tolist() == [1, 2, 3, 41, 42, 43]
    assert np.flatnonzero(c).tolist() == np.flatnonzero(d).tolist() == [2]
    assert r[-1] == 2
    backup = cost_target(torch.from_numpy(c), torch.from_numpy(d), torch.ones(6), .9)
    assert torch.allclose(backup, torch.tensor([.9, .9, 1., .9, .9, .9]))
    with pytest.raises(ValueError):
        cost_target(torch.tensor(c)[:, None], torch.tensor(d), torch.ones(6), .9)


def test_ood_cost_has_grad_only_for_online_critic_with_legal_actions():
    model = small()
    s, a, selected = model.candidate_set(torch.randn(32, 35))
    assert a.shape == (160, 2) and (a.abs() <= 1.).all()
    assert 0 < selected.sum() < len(selected)
    loss = model.ood_cost(s, a, selected)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0. for p in model.cost_critic.parameters())
    for module in (model.actor, model.vae, model.cost_critic_old):
        assert all(p.grad is None for p in module.parameters())


def test_updates_and_checkpoint_reproduction(tmp_path):
    model = small()
    s, ns = torch.randn(16, 35), torch.randn(16, 35)
    a, r, c, terminal = torch.zeros(16, 2), torch.randn(16), torch.ones(16), torch.ones(16)
    for loss, stats in (model.vae_loss(s, a), model.critic_loss(s, ns, a, r, terminal),
                        model.cost_critic_loss(s, ns, a, c, terminal), model.actor_loss(s)):
        assert torch.isfinite(loss).item() and all(np.isfinite(v) for v in stats.values())
    model.sync_weight()
    obs = {"state": s[0].numpy()}
    expected = DeploymentPolicy(model.actor, "cpu").act(obs)
    path = tmp_path/"cpq.pt"
    save_policy(path, model, {"a_hidden_sizes": [8]}, 1, {})
    restored = load_policy(path, "cpu")
    np.testing.assert_array_equal(restored.act(obs), expected)
    assert expected.shape == (2,) and np.all(np.abs(expected) <= 1.)
    with pytest.raises(FileExistsError):
        save_policy(path, model, {"a_hidden_sizes": [8]}, 1, {})
    with pytest.raises(ValueError):
        restored.act({"state": np.zeros(36)})
