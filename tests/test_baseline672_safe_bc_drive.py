import numpy as np
import pytest
import torch

from baseline672.drive import drive_episode
from baseline672.safe_bc import SCHEMA, SafeBC, load_policy, pairs


def episode():
    return dict(states=np.arange(105, dtype=np.float32).reshape(3, 35) / 100.,
                actions=np.array([[.1, .2], [-.2, -.1]], dtype=np.float32),
                costs=np.zeros(2), truncated=np.zeros(2, dtype=bool), outcome="success")


def test_supervise_only_real_actions_and_reload(tmp_path):
    ep = episode()
    x, a = pairs([ep])
    np.testing.assert_array_equal(x.numpy(), ep["states"][:-1])
    np.testing.assert_array_equal(a.numpy(), ep["actions"])
    torch.manual_seed(1)
    net = SafeBC()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    obs = {"state": ep["states"][0]}
    before = net.act(obs).copy()
    net.loss(x, a).backward()
    opt.step()
    assert not np.allclose(before, net.act(obs))
    ck = tmp_path / "bc.pt"
    torch.save({"schema": SCHEMA, "weights": net.state_dict()}, ck)
    np.testing.assert_array_equal(net.act(obs), load_policy(ck).act(obs))


@pytest.mark.parametrize("change", [dict(outcome="collision"), dict(costs=[0, 1]),
                                    dict(truncated=[False, True])])
def test_bc_validates_safe_filter(change):
    with pytest.raises(ValueError):
        pairs([dict(episode(), **change)])


def test_rollout_order_and_bounded_actions():
    log = []
    class Env:
        def reset(self, force_seed):
            log.append(("reset_env", force_seed))
            return {"state": 7}
        def step(self, a):
            log.append(("step", tuple(a)))
            return {"state": 8}, 0., True, {"cost": 1., "crash_vehicle": True, "arrive_dest": True,
                                            "raw_action": a}
    class Agent:
        def reset(self):
            log.append("reset_agent")
        def act(self, obs):
            log.append(("act", obs["state"]))
            return np.array([.25, -.5])
        def observe(self, a, r, c):
            log.append(("observe", r, c))
    row = drive_episode(Env(), Agent(), 9, 10)
    assert row["outcome"] == "collision" and row["steps"] == 1
    assert log == [("reset_env", 9), "reset_agent", ("act", 7), ("step", (.25, -.5)), ("observe", 0., 1.)]
