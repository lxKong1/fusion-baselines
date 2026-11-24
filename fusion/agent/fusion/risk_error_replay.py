"""Critical-braking selection for the auxiliary FUSION imitation loss."""
import numpy as np
from torch.utils.data import Dataset

from .cpa_risk import CpaRiskConfig, cpa_risk_for_states

def _contiguous_true_segments(mask):
    """Return half-open intervals for every contiguous true segment."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("segment mask must be one-dimensional")
    previous = np.concatenate((np.asarray([False]), mask[:-1]))
    following = np.concatenate((mask[1:], np.asarray([False])))
    starts = np.flatnonzero(mask & ~previous)
    stops = np.flatnonzero(mask & ~following) + 1
    return list(zip(starts.tolist(), stops.tolist()))


class SeverityCpaEventWindowDataset(Dataset):
    """Mine one first-crossing window per critical braking-response event.

    The per-timestep score is

    ``([ -a_long ]_+ + epsilon)**gamma * (1 + risk_lambda * CPA_risk)``.

    Consecutive timesteps above ``score_threshold`` form one event.  The first
    qualifying timestep is the event anchor, preserving the pre-response
    history used by the validated action-onset replay.  Event scores are used
    only for discovery; every resulting window remains equally weighted by
    the auxiliary action-loss DataLoader.
    """

    def __init__(
        self,
        source_dataset,
        pre_steps=10,
        post_steps=10,
        score_threshold=0.252,
        epsilon=1e-3,
        severity_gamma=2.0,
        risk_lambda=0.2,
        risk_config=CpaRiskConfig(),
    ):
        self.source_dataset = source_dataset
        self.pre_steps = int(pre_steps)
        self.post_steps = int(post_steps)
        self.score_threshold = float(score_threshold)
        self.epsilon = float(epsilon)
        self.severity_gamma = float(severity_gamma)
        self.risk_lambda = float(risk_lambda)
        self.risk_config = risk_config.validate()

        if self.pre_steps < 0 or self.post_steps < 1:
            raise ValueError(
                "event windows require non-negative pre_steps and positive "
                "post_steps"
            )
        if (
            self.pre_steps + self.post_steps
            != int(source_dataset.context_len)
        ):
            raise ValueError(
                "pre_steps + post_steps must equal context_len ({} + {} != {})".format(
                    self.pre_steps,
                    self.post_steps,
                    source_dataset.context_len,
                )
            )
        if int(source_dataset.context_len) != self.risk_config.context_len:
            raise ValueError(
                "CPA context_len {} must equal dataset context_len {}".format(
                    self.risk_config.context_len,
                    source_dataset.context_len,
                )
            )
        if self.score_threshold <= 0.0:
            raise ValueError("score_threshold must be positive")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        if self.severity_gamma <= 0.0:
            raise ValueError("severity_gamma must be positive")
        if self.risk_lambda < 0.0:
            raise ValueError("risk_lambda cannot be negative")

        self.events = []
        event_trajectory_ids = set()
        hard_events = 0
        mild_events = 0
        risk_only_events = 0
        qualifying_steps = 0
        anchor_offsets_from_action_onset = []
        peak_offsets_from_anchor = []

        for trajectory_index, trajectory in enumerate(
            source_dataset.trajectories
        ):
            states = np.asarray(trajectory["state"], dtype=np.float32)
            actions = np.asarray(trajectory["actions"], dtype=np.float64)
            if actions.shape != (len(states), 2):
                raise ValueError(
                    "trajectory actions must have shape ({}, 2), got {}".format(
                        len(states), actions.shape
                    )
                )
            if not np.all(np.isfinite(actions)):
                raise ValueError("trajectory actions contain non-finite values")
            if np.max(np.abs(actions[:, 1])) > 1.0 + 1e-6:
                raise ValueError(
                    "longitudinal expert actions must lie in [-1, 1]"
                )

            timestep_risk = cpa_risk_for_states(
                states, self.risk_config
            )["timestep_risk"]
            braking_strength = np.maximum(-actions[:, 1], 0.0)
            scores = np.power(
                braking_strength + self.epsilon,
                self.severity_gamma,
            ) * (1.0 + self.risk_lambda * timestep_risk)
            if not np.all(np.isfinite(scores)):
                raise ValueError("critical-event scores are non-finite")

            qualifying = scores > self.score_threshold
            qualifying_steps += int(np.sum(qualifying))
            action_braking_segments = _contiguous_true_segments(
                actions[:, 1] < -0.25
            )
            for event_start, event_stop in _contiguous_true_segments(
                qualifying
            ):
                anchor = int(event_start)
                peak = anchor + int(np.argmax(scores[event_start:event_stop]))
                window_start = max(0, anchor - self.pre_steps)
                event_braking_strength = braking_strength[
                    event_start:event_stop
                ]
                maximum_braking = float(np.max(event_braking_strength))
                if maximum_braking >= 0.75:
                    event_class = "hard_brake"
                    hard_events += 1
                elif maximum_braking > 0.0:
                    event_class = "mild_brake"
                    mild_events += 1
                else:
                    event_class = "risk_only"
                    risk_only_events += 1

                containing_action_onsets = [
                    start
                    for start, stop in action_braking_segments
                    if start <= anchor < stop
                ]
                action_onset = (
                    int(containing_action_onsets[0])
                    if len(containing_action_onsets) == 1
                    else None
                )
                if action_onset is not None:
                    anchor_offsets_from_action_onset.append(
                        anchor - action_onset
                    )
                peak_offsets_from_anchor.append(peak - anchor)

                event = {
                    "trajectory_index": int(trajectory_index),
                    "trajectory_id": int(
                        trajectory.get("trajectory_id", trajectory_index)
                    ),
                    "event_start": anchor,
                    "event_stop": int(event_stop),
                    "anchor": anchor,
                    "peak": peak,
                    "start": window_start,
                    "anchor_position": anchor - window_start,
                    "action_onset": action_onset,
                    "event_class": event_class,
                    "anchor_score": float(scores[anchor]),
                    "peak_score": float(scores[peak]),
                    "anchor_risk": float(timestep_risk[anchor]),
                    "peak_risk": float(timestep_risk[peak]),
                    "maximum_braking": maximum_braking,
                }
                self.events.append(event)
                event_trajectory_ids.add(event["trajectory_id"])

        if not self.events:
            raise ValueError(
                "no critical braking-response events found above score "
                "threshold {}".format(self.score_threshold)
            )

        self.statistics = {
            "brake_window_selector": "severity_cpa_first_crossing",
            "brake_event_score_threshold": self.score_threshold,
            "brake_event_score_epsilon": self.epsilon,
            "brake_event_score_gamma": self.severity_gamma,
            "brake_event_score_risk_lambda": self.risk_lambda,
            "brake_window_pre_steps": self.pre_steps,
            "brake_window_post_steps": self.post_steps,
            "brake_event_windows": len(self.events),
            "brake_event_trajectories": len(event_trajectory_ids),
            "brake_event_qualifying_steps": qualifying_steps,
            "brake_event_hard_windows": hard_events,
            "brake_event_mild_windows": mild_events,
            "brake_event_risk_only_windows": risk_only_events,
            "brake_event_mean_anchor_offset_from_action_onset": (
                float(np.mean(anchor_offsets_from_action_onset))
                if anchor_offsets_from_action_onset
                else float("nan")
            ),
            "brake_event_max_anchor_offset_from_action_onset": (
                int(np.max(anchor_offsets_from_action_onset))
                if anchor_offsets_from_action_onset
                else -1
            ),
            "brake_event_mean_peak_offset_from_anchor": float(
                np.mean(peak_offsets_from_anchor)
            ),
        }

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        event = self.events[int(index)]
        return self.source_dataset.window_at(
            event["trajectory_index"], event["start"]
        )
