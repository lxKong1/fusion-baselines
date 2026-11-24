"""Predict actions and return estimates for the current observation history."""


def predict_action_and_values(
    model,
    timesteps,
    states,
    lidar,
    actions,
    rewards_to_go,
    costs_to_go,
    bev=None,
    predict_values=True,
    return_diagnostics=False,
):
    """Autoregressively predict an action and optional next-step values.

    The action head is conditioned on ``(C_t, R_t, S_t)``. The value heads
    occur after the action token, so a second pass is required to condition
    them on the generated action rather than the zero action placeholder.
    When requested, diagnostics expose the action-conditioned 35-D next-state
    prediction from that same second pass without changing model execution.
    """
    model_states = [states, lidar]
    if getattr(model, "use_bev", False):
        if bev is None:
            raise ValueError("Multimodal FUSION rollout requires a BEV tensor")
        model_states.append(bev)
    elif bev is not None:
        raise ValueError("Vector-only FUSION rollout received an unexpected BEV")

    _, action_predictions, _, _ = model.forward(
        timesteps,
        model_states,
        actions,
        rewards_to_go,
        costs_to_go,
    )
    action = action_predictions[:, -1]

    if not predict_values and not return_diagnostics:
        return action, None, None

    conditioned_actions = actions.clone()
    conditioned_actions[:, -1] = action
    state_predictions, _, return_predictions, cost_predictions = model.forward(
        timesteps,
        model_states,
        conditioned_actions,
        rewards_to_go,
        costs_to_go,
    )
    next_rtg = return_predictions[:, -1] if predict_values else None
    next_ctg = cost_predictions[:, -1] if predict_values else None
    if return_diagnostics:
        diagnostics = {
            "predicted_next_state": state_predictions[:, -1],
        }
        return action, next_rtg, next_ctg, diagnostics
    return action, next_rtg, next_ctg
