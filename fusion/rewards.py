"""Canonical task reward shared by offline labels and MetaDrive rollouts.

The public labels store MetaDrive's dense ``step_reward`` before MetaDrive
applies terminal reward overrides.  Consequently, directly summing that field
omits the destination reward.  This module reconstructs task reward in memory,
caps the positive speed incentive at 30 km/h, and supplies the same constants
to the online environment.  Raw dataset files are never modified.
"""

from collections.abc import Mapping

import numpy as np

from fusion.costs import safety_violation_from_record


DRIVING_REWARD_WEIGHT = 1.0
SPEED_REWARD_WEIGHT = 0.1
DESIRED_SPEED_KM_H = 30.0
GOAL_REWARD = 50.0

# MetaDrive's built-in speed term is linear and cannot express saturation.  It
# is disabled in the simulator configuration and added explicitly by the
# canonical reward function below.
METADRIVE_BASE_SPEED_REWARD_WEIGHT = 0.0

# The released labels were collected with MetaDrive's defaults.  Their stored
# dense reward contains 0.1 * velocity / 80.  We remove that uncapped term
# before adding the capped research reward, so the source reward is not counted
# twice.
SOURCE_SPEED_REWARD_WEIGHT = 0.1
SOURCE_MAX_SPEED_KM_H = 80.0

# Public labels and locally collected human demonstrations store different
# dense-reward sources. Public ``step_reward`` values include MetaDrive's
# uncapped speed term, whereas the current human collector stores progress only
# and records the canonical capped reward separately. Keeping the protocol
# explicit prevents subtracting a speed term that is not present.
RELEASED_SPEED80_REWARD_PROTOCOL = "released_metadrive_speed80_embedded"
PROGRESS_ONLY_REWARD_PROTOCOL = "progress_only_canonical_speed30"

REWARD_DEFINITION = "progress_capped_speed30_goal50_terminal_outcome"


def _scalar_float(value, name):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("{} must be scalar, got shape {}".format(name, array.shape))
    result = float(array.reshape(-1)[0])
    if not np.isfinite(result):
        raise ValueError("{} must be finite".format(name))
    return result


def arrives_at_destination(record):
    """Return whether one offline/online record reports task completion."""
    if not isinstance(record, Mapping):
        raise TypeError("reward records must be mappings")
    value = np.asarray(record.get("arrive_dest", False))
    if value.size != 1:
        raise ValueError(
            "arrive_dest must be scalar, got shape {}".format(value.shape)
        )
    return bool(value.reshape(-1)[0])


def source_speed_reward(record):
    """Return the explicit speed term embedded in a released label."""
    if not isinstance(record, Mapping):
        raise TypeError("reward records must be mappings")
    velocity = _scalar_float(record["velocity"], "velocity")
    return SOURCE_SPEED_REWARD_WEIGHT * velocity / SOURCE_MAX_SPEED_KM_H


def capped_speed_reward_from_velocity(velocity):
    """Return ``0.1 * min(velocity, 30) / 80`` in reward units."""
    velocity = _scalar_float(velocity, "velocity")
    return (
        SPEED_REWARD_WEIGHT
        * min(velocity, DESIRED_SPEED_KM_H)
        / SOURCE_MAX_SPEED_KM_H
    )


def capped_speed_reward(record):
    """Return the desired capped speed term for one released label."""
    if not isinstance(record, Mapping):
        raise TypeError("reward records must be mappings")
    return capped_speed_reward_from_velocity(record["velocity"])


def reward_source_protocol(record):
    """Return the dense-reward storage protocol for one trajectory record.

    Released FUSION labels predate explicit protocol metadata and therefore
    use the historical speed-80 protocol by default. Human records collected
    by this repository carry either an explicit protocol or the backwards-
    compatible ``human_demonstration`` marker.
    """
    if not isinstance(record, Mapping):
        raise TypeError("reward records must be mappings")
    protocol = record.get("reward_source_protocol")
    if protocol is None:
        return (
            PROGRESS_ONLY_REWARD_PROTOCOL
            if bool(record.get("human_demonstration", False))
            else RELEASED_SPEED80_REWARD_PROTOCOL
        )
    if protocol not in (
        RELEASED_SPEED80_REWARD_PROTOCOL,
        PROGRESS_ONLY_REWARD_PROTOCOL,
    ):
        raise ValueError("unknown reward source protocol: {!r}".format(protocol))
    return protocol


def progress_reward_from_record(record):
    """Recover longitudinal progress under the record's source protocol.

    The released environment used ``driving_reward=1``,
    ``speed_reward=0.1``, ``vehicle.max_speed=80``, and no lateral multiplier.
    Public trajectories overwhelmingly follow the positive route direction;
    this subtraction isolates the stored longitudinal progress signal before
    the desired capped speed term is added separately. Locally collected human
    records already store progress-only ``step_reward`` and must not undergo
    that subtraction.
    """
    if not isinstance(record, Mapping):
        raise TypeError("reward records must be mappings")
    dense_reward = _scalar_float(record["step_reward"], "step_reward")
    if reward_source_protocol(record) == PROGRESS_ONLY_REWARD_PROTOCOL:
        return dense_reward
    return dense_reward - source_speed_reward(record)


def task_reward_from_record(record):
    """Return the canonical task reward for one retained transition.

    Severe safety events take precedence over a simultaneous destination flag.
    This matches the absorbing binary-cost protocol: their terminal task reward
    is zero and their safety cost is one.  A clean destination transition is
    assigned the complete terminal reward of 50; all other transitions retain
    longitudinal progress plus a speed reward that saturates at 30 km/h.
    """
    if safety_violation_from_record(record):
        return 0.0
    if arrives_at_destination(record):
        return GOAL_REWARD
    return progress_reward_from_record(record) + capped_speed_reward(record)


def task_rewards(records):
    """Return shape ``(T, 1)`` rewards without mutating source records."""
    return np.asarray(
        [task_reward_from_record(record) for record in records],
        dtype=np.float32,
    ).reshape(-1, 1)


def reward_protocol_metadata():
    """Return JSON/W&B-safe metadata for the active reward protocol."""
    return {
        "reward_definition": REWARD_DEFINITION,
        "driving_reward_weight": DRIVING_REWARD_WEIGHT,
        "speed_reward_weight": SPEED_REWARD_WEIGHT,
        "desired_speed_km_h": DESIRED_SPEED_KM_H,
        "metadrive_base_speed_reward_weight": (
            METADRIVE_BASE_SPEED_REWARD_WEIGHT
        ),
        "goal_reward": GOAL_REWARD,
        "source_speed_reward_weight": SOURCE_SPEED_REWARD_WEIGHT,
        "source_max_speed_km_h": SOURCE_MAX_SPEED_KM_H,
        "released_reward_source_protocol": RELEASED_SPEED80_REWARD_PROTOCOL,
        "human_reward_source_protocol": PROGRESS_ONLY_REWARD_PROTOCOL,
    }
