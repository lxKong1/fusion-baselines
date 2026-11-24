"""Canonical severe-safety cost shared by offline data and MetaDrive."""

from collections.abc import Mapping

import numpy as np


COLLISION_KEYS = (
    "crash",
    "crash_vehicle",
    "crash_object",
    "crash_building",
)
OUT_OF_ROAD_KEY = "out_of_road"
COST_DEFINITION = "binary_collision_or_offroad_first_event_terminal"


def _scalar_bool(value, name):
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("{} must be scalar, got shape {}".format(name, array.shape))
    return bool(array.reshape(-1)[0])


def collision_from_record(record):
    """Return whether a label/info mapping reports any physical collision."""
    if not isinstance(record, Mapping):
        raise TypeError("safety records must be mappings")
    return any(
        _scalar_bool(record.get(key, False), key) for key in COLLISION_KEYS
    )


def out_of_road_from_record(record):
    """Return whether a label/info mapping reports leaving the drivable area."""
    if not isinstance(record, Mapping):
        raise TypeError("safety records must be mappings")
    return _scalar_bool(record.get(OUT_OF_ROAD_KEY, False), OUT_OF_ROAD_KEY)


def safety_violation_from_record(record):
    """Return the canonical severe-safety event indicator for one step."""
    return collision_from_record(record) or out_of_road_from_record(record)


def binary_safety_cost(collision, out_of_road):
    """Return one for collision OR off-road and zero otherwise.

    Logical OR is intentional: simultaneous event flags still define one
    terminal safety violation rather than an additive cost of two.
    """
    return float(
        _scalar_bool(collision, "collision")
        or _scalar_bool(out_of_road, "out_of_road")
    )


def safety_cost_from_record(record):
    """Return the canonical binary cost for one offline or online record."""
    return binary_safety_cost(
        collision_from_record(record),
        out_of_road_from_record(record),
    )


def terminate_after_safety_event(base_done, record):
    """Combine simulator termination with the absorbing safety-event rule."""
    return _scalar_bool(base_done, "base_done") or safety_violation_from_record(
        record
    )


def safety_event_flags(records):
    """Return one boolean severe-event flag per record."""
    return np.asarray(
        [safety_violation_from_record(record) for record in records],
        dtype=bool,
    )


def safety_costs(records):
    """Return shape ``(T, 1)`` binary costs derived only from event flags."""
    return safety_event_flags(records).astype(np.float32).reshape(-1, 1)


def first_safety_event_index(records):
    """Return the first severe-event index, or ``None`` for a safe trajectory."""
    indices = np.flatnonzero(safety_event_flags(records))
    return None if not len(indices) else int(indices[0])


def truncate_at_first_safety_event(records):
    """Copy a trajectory prefix through and including its first severe event.

    The input sequence and its mappings are never mutated. Keeping the event
    transition preserves supervision for the state/action that led to danger;
    all recovery and repeated-event records after it are excluded.
    """
    records = list(records)
    first_event = first_safety_event_index(records)
    stop = len(records) if first_event is None else first_event + 1
    return records[:stop], first_event
