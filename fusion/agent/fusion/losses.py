"""Loss functions for the Causal Ensemble World Model (CEWM)."""

import torch
import torch.nn.functional as F


def _masked_mse(predictions, targets, mask, name):
    """Compute MSE over valid sequence positions without flattening time early."""
    if predictions.shape != targets.shape:
        raise ValueError(
            "{} prediction/target shapes differ: {} != {}".format(
                name, tuple(predictions.shape), tuple(targets.shape)
            )
        )
    if predictions.ndim != 3:
        raise ValueError(
            "{} tensors must have shape (batch, time, features), got {}".format(
                name, tuple(predictions.shape)
            )
        )
    if mask.ndim != 2 or tuple(mask.shape) != tuple(predictions.shape[:2]):
        raise ValueError(
            "{} mask must have shape {}, got {}".format(
                name, tuple(predictions.shape[:2]), tuple(mask.shape)
            )
        )

    valid = mask.to(dtype=torch.bool)
    if not torch.any(valid):
        raise ValueError("{} loss has no valid sequence positions".format(name))

    return F.mse_loss(predictions[valid], targets[valid], reduction="mean")


def compute_action_loss(action_predictions, action_targets, trajectory_mask):
    """Return the masked current-step behavior-cloning objective."""
    return _masked_mse(
        action_predictions,
        action_targets,
        trajectory_mask,
        "action",
    )


def mix_action_losses(uniform_loss, brake_window_loss, brake_window_ratio):
    """Convexly mix natural-distribution and brake-window action losses."""
    ratio = float(brake_window_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("brake_window_ratio must lie in [0, 1]")
    if uniform_loss.ndim != 0 or brake_window_loss.ndim != 0:
        raise ValueError("action losses must be scalar tensors")
    return (1.0 - ratio) * uniform_loss + ratio * brake_window_loss


def compute_cewm_losses(
    state_predictions,
    action_predictions,
    return_predictions,
    cost_predictions,
    state_targets,
    action_targets,
    return_targets,
    cost_targets,
    trajectory_mask,
    include_dynamics=True,
    include_value=True,
):
    """Compute the CEWM objective with explicit temporal alignment.

    The policy predicts the action at the current step. The dynamics, reward
    critic, and cost critic heads are conditioned on the current action token
    and predict targets at the following step. A transition is valid only when
    both of its endpoints are present in ``trajectory_mask``.
    """
    if trajectory_mask.ndim != 2:
        raise ValueError(
            "trajectory_mask must have shape (batch, time), got {}".format(
                tuple(trajectory_mask.shape)
            )
        )

    batch_time = tuple(trajectory_mask.shape)
    named_sequences = {
        "state_predictions": state_predictions,
        "action_predictions": action_predictions,
        "return_predictions": return_predictions,
        "cost_predictions": cost_predictions,
        "state_targets": state_targets,
        "action_targets": action_targets,
        "return_targets": return_targets,
        "cost_targets": cost_targets,
    }
    for name, tensor in named_sequences.items():
        if tensor.ndim != 3 or tuple(tensor.shape[:2]) != batch_time:
            raise ValueError(
                "{} must start with shape {}, got {}".format(
                    name, batch_time, tuple(tensor.shape)
                )
            )

    losses = {
        "action": compute_action_loss(
            action_predictions,
            action_targets,
            trajectory_mask,
        )
    }

    if include_dynamics or include_value:
        if batch_time[1] < 2:
            raise ValueError(
                "At least two timesteps are required for next-step CEWM losses"
            )
        valid_steps = trajectory_mask.to(dtype=torch.bool)
        transition_mask = valid_steps[:, :-1] & valid_steps[:, 1:]

    if include_dynamics:
        losses["dynamics"] = _masked_mse(
            state_predictions[:, :-1],
            state_targets[:, 1:],
            transition_mask,
            "dynamics",
        )

    if include_value:
        losses["rtg"] = _masked_mse(
            return_predictions[:, :-1],
            return_targets[:, 1:],
            transition_mask,
            "reward-to-go",
        )
        losses["ctg"] = _masked_mse(
            cost_predictions[:, :-1],
            cost_targets[:, 1:],
            transition_mask,
            "cost-to-go",
        )

    return losses


def ensure_finite_losses(losses):
    """Fail before optimization if any scalar objective is NaN or infinite."""
    non_finite = [
        name for name, loss in losses.items() if not torch.isfinite(loss).all()
    ]
    if non_finite:
        raise FloatingPointError(
            "Non-finite CEWM loss(es): {}".format(", ".join(non_finite))
        )
