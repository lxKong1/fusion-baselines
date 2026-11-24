"""Closest-point-of-approach features for baseline replay window selection."""
from dataclasses import dataclass, asdict

import numpy as np

EGO_NAVIGATION_DIM = 19
SURROUNDING_VEHICLE_COUNT = 4
SURROUNDING_VEHICLE_DIM = 4
FUSION_STATE_DIM = (
    EGO_NAVIGATION_DIM
    + SURROUNDING_VEHICLE_COUNT * SURROUNDING_VEHICLE_DIM
)


@dataclass(frozen=True)
class CpaRiskConfig:
    """Physical constants and audit choices for CPA risk."""

    context_len: int = 20
    horizon_seconds: float = 3.0
    time_scale_seconds: float = 2.0
    distance_scale_meters: float = 2.0
    lidar_distance_meters: float = 50.0
    vehicle_max_speed_kmh: float = 80.0
    relative_speed_epsilon_mps: float = 1e-3
    error_neighborhood_radius: int = 2
    mild_brake_threshold: float = -0.05
    # Keyboard collection produces two braking levels (-0.5 and -1.0).
    # The midpoint keeps them in distinct mild/hard audit groups.
    hard_brake_threshold: float = -0.75

    def validate(self):
        if self.context_len < 1:
            raise ValueError("context_len must be positive")
        positive = {
            "horizon_seconds": self.horizon_seconds,
            "time_scale_seconds": self.time_scale_seconds,
            "distance_scale_meters": self.distance_scale_meters,
            "lidar_distance_meters": self.lidar_distance_meters,
            "vehicle_max_speed_kmh": self.vehicle_max_speed_kmh,
            "relative_speed_epsilon_mps": self.relative_speed_epsilon_mps,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.error_neighborhood_radius < 0:
            raise ValueError("error_neighborhood_radius cannot be negative")
        if not -1.0 <= self.hard_brake_threshold < self.mild_brake_threshold < 0:
            raise ValueError(
                "brake thresholds must satisfy -1 <= hard < mild < 0"
            )
        return self

    def to_dict(self):
        return asdict(self)


def _state_array(states):
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
    if states.ndim != 2 or states.shape[1] != FUSION_STATE_DIM:
        raise ValueError(
            "states must have shape (time, {}), got {}".format(
                FUSION_STATE_DIM, states.shape
            )
        )
    if not np.all(np.isfinite(states)):
        raise ValueError("states contain non-finite values")
    return states


def decode_surrounding_vehicles(states, config=CpaRiskConfig()):
    """Decode four nearest vehicles from one or more normalized states.

    Returns ``(relative_position_m, relative_velocity_mps, present)`` with
    leading time dimension.  MetaDrive pads a missing surrounding-vehicle
    slot with four exact zeros; it is masked before normalized values are
    converted to physical units.
    """

    config.validate()
    states = _state_array(states)
    encoded = states[:, EGO_NAVIGATION_DIM:].reshape(
        len(states), SURROUNDING_VEHICLE_COUNT, SURROUNDING_VEHICLE_DIM
    )
    present = np.any(encoded != 0.0, axis=-1)

    positions = (
        (2.0 * encoded[..., :2] - 1.0) * config.lidar_distance_meters
    )
    relative_velocity = (
        (2.0 * encoded[..., 2:] - 1.0)
        * config.vehicle_max_speed_kmh
        / 3.6
    )

    # Padding would otherwise decode to a fictitious vehicle at (-50, -50)
    # moving at (-80, -80) km/h.
    positions = np.where(present[..., None], positions, 0.0)
    relative_velocity = np.where(
        present[..., None], relative_velocity, 0.0
    )
    return positions, relative_velocity, present


def cpa_risk_for_states(states, config=CpaRiskConfig()):
    """Return per-timestep, per-neighbour TCPA, DCPA, and risk arrays."""

    config.validate()
    positions, relative_velocity, present = decode_surrounding_vehicles(
        states, config
    )
    speed_squared = np.sum(relative_velocity ** 2, axis=-1)
    moving = speed_squared > config.relative_speed_epsilon_mps ** 2

    dot_product = np.sum(positions * relative_velocity, axis=-1)
    tcpa = np.zeros_like(speed_squared)
    np.divide(
        -dot_product,
        speed_squared,
        out=tcpa,
        where=moving,
    )
    cpa_position = positions + tcpa[..., None] * relative_velocity
    dcpa = np.linalg.norm(cpa_position, axis=-1)

    valid = (
        present
        & moving
        & (tcpa > 0.0)
        & (tcpa < config.horizon_seconds)
    )
    risk = np.zeros_like(tcpa)
    risk[valid] = np.exp(
        -tcpa[valid] / config.time_scale_seconds
        -dcpa[valid] / config.distance_scale_meters
    )

    tcpa = np.where(valid, tcpa, np.nan)
    dcpa = np.where(valid, dcpa, np.nan)
    best_slot = np.argmax(risk, axis=1).astype(np.int64)
    timestep_risk = np.max(risk, axis=1)
    best_slot[timestep_risk <= 0.0] = -1
    return {
        "positions_m": positions,
        "relative_velocity_mps": relative_velocity,
        "present": present,
        "tcpa_seconds": tcpa,
        "dcpa_meters": dcpa,
        "risk_by_vehicle": risk,
        "timestep_risk": timestep_risk,
        "best_slot": best_slot,
    }
