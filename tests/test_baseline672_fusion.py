import numpy as np
import pytest
import torch

from baseline672.fusion_policy import (FORMAT, NET, Fusion_Structure, Policy, Windows,
                                     bisimulation_update, load_policy)


def episode(n, failed=True):
    return dict(actions=np.zeros((n, 2), np.float32),
                states=np.broadcast_to(np.arange(n + 1, dtype=np.float32)[:, None], (n + 1, 35)),
                lidars=np.zeros((n + 1, 240), np.float32),
                bevs=np.zeros((n + 1, 5, 84, 84), np.uint8),
                rewards=np.arange(1, n + 1, dtype=np.float32),
                costs=np.asarray([0] * (n - 1) + [int(failed)], np.float32))


def test_window_has_current_observations_actual_actions_and_future_costs():
    ds = Windows([episode(3), episode(4, False)])
    t, obs, a, r, c, m = ds.window_at(0, 1)
    assert m.sum() == 2 and m[:2].all() and not m[2:].any()
    np.testing.assert_array_equal(t[:2], [1, 2])
    np.testing.assert_array_equal(obs[0][:3, 0], [1, 2, 0])
    assert obs[2].dtype == torch.uint8
    np.testing.assert_allclose(r[:3, 0], np.asarray([5, 3, 0]) / 300)
    np.testing.assert_allclose(c[:3, 0], [.1, .1, 0])
    assert not ds.window_at(1, 0)[4].any()
    with pytest.raises(IndexError):
        ds.window_at(0, 3)


class RecordingNet(torch.nn.Module):
    use_bev = True

    def forward(self, t, obs, a, r, c):
        self.inputs = t, obs, a, r, c
        return obs[0], torch.zeros_like(a), r, c


def test_rollout_uses_only_executed_past_actions_and_observed_rewards():
    model = RecordingNet()
    p = Policy(model, "cpu")
    obs = dict(state=np.ones(35), lidar=np.ones(240), bev=np.full((5, 84, 84), 255, np.uint8))
    p.act(obs)
    with pytest.raises(RuntimeError):
        p.act(obs)
    p.observe([.25, .7], reward=6, cost=0)
    p.act(obs)
    t, fields, a, r, c = model.inputs
    np.testing.assert_array_equal(t[0, -2:], [0, 1])
    np.testing.assert_allclose(a[0, -2:], [[.25, .7], [0., 0.]])
    np.testing.assert_allclose(r[0, -2:, 0], [200 / 300, 194 / 300])
    assert not c.any()
    assert not fields[2][0, :-2].any() and (fields[2][0, -2:] == 1).all()
    p.reset()
    p.act(obs)
    assert not model.inputs[2].any()
    with pytest.raises(ValueError):
        p.observe([np.nan, 1.], 0., 0.)


def test_save_reload_selects_identical_action(tmp_path):
    model = Fusion_Structure(**NET)
    obs = dict(state=np.ones(35), lidar=np.ones(240), bev=np.zeros((5, 84, 84), np.float32))
    path = tmp_path / "trained.pt"
    torch.save(dict(format=FORMAT, spec=NET, model=model.state_dict()), path)
    p = load_policy(path, "cpu")
    np.testing.assert_array_equal(p.act(obs), Policy(model, "cpu").act(obs))


def test_bisim_updates_active_encoder():
    torch.manual_seed(42)
    model = Fusion_Structure(**NET)
    # The separate tokenizer optimizer never updates the action-output head.
    opt = torch.optim.AdamW(model.embed_state.parameters(), lr=.001)
    ds = Windows([episode(7), episode(11, False)])
    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
    before = model.embed_state.weight.detach().clone()
    bisimulation_update(model, opt, batch)
    assert torch.isfinite(model.embed_state.weight).all()
    assert not torch.equal(before, model.embed_state.weight)
