"""Tests for the circuit geometry.

The closure tests are the important ones. A circuit that does not close builds a
map whose end never meets its start, which MetaDrive accepts silently -- the
failure only shows up later as a car that cannot complete a lap. So every
shipped circuit is asserted to close here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drive_rl.circuits import (
    CIRCUITS,
    LEFT,
    RIGHT,
    Block,
    Circuit,
    CircuitNotClosedError,
    ReferenceLine,
    get_circuit,
    left,
    right,
    sample_centerline,
    straight,
    trace,
    validate_circuit,
)


def test_all_shipped_circuits_close():
    for name, circuit in CIRCUITS.items():
        validate_circuit(circuit)
        geometry = trace(circuit)
        assert abs(abs(geometry.total_turn_deg) - 360.0) <= 0.5, name
        assert math.dist(geometry.end, (0.0, 0.0)) <= 2.0, name


def test_every_circuit_has_corners_and_positive_length():
    for name, circuit in CIRCUITS.items():
        assert circuit.corner_count >= 8, name
        assert circuit.length_m > 1000.0, name
        assert circuit.lane_width >= 8.0, name


def test_direction_signs_are_opposite():
    assert right(50.0, 90.0).signed_angle < 0.0
    assert left(50.0, 90.0).signed_angle > 0.0


def test_block_geometry_relations():
    b = Block("curve", length=10.0, radius=100.0, angle=90.0, direction=RIGHT)
    assert math.isclose(b.arc_length, 100.0 * math.radians(90.0))
    assert math.isclose(b.travel_length, b.arc_length + 10.0)
    assert straight(25.0).arc_length == 0.0
    assert straight(25.0).travel_length == 25.0


def test_curve_rejects_invalid_radius():
    with pytest.raises(ValueError):
        Block("curve", length=0.0, radius=0.0, angle=90.0, direction=LEFT)


def test_straight_rejects_negative_length():
    with pytest.raises(ValueError):
        Block("straight", length=-5.0)


def test_a_square_is_a_valid_circuit():
    square = Circuit(
        "square",
        (
            straight(100.0),
            right(50.0, 90.0),
            straight(100.0),
            right(50.0, 90.0),
            straight(100.0),
            right(50.0, 90.0),
            straight(100.0),
            right(50.0, 90.0),
        ),
        lane_width=10.0,
    )
    geometry = validate_circuit(square)
    assert abs(geometry.total_turn_deg + 360.0) < 1e-6
    assert math.dist(geometry.end, (0.0, 0.0)) < 1e-6


def test_open_path_is_rejected():
    # Only three corners: the net turn is nowhere near a full revolution.
    broken = Circuit("broken", (straight(100.0), right(50.0, 90.0), straight(100.0)))
    with pytest.raises(CircuitNotClosedError):
        validate_circuit(broken)


def test_get_circuit_lookup():
    assert get_circuit("SUZUKA").name == "suzuka"
    assert get_circuit("monaco").name == "monaco"
    with pytest.raises(KeyError):
        get_circuit("nurburgring")


def test_reference_line_length_matches_analytic_lap():
    for name, circuit in CIRCUITS.items():
        points, arc = sample_centerline(circuit, spacing=2.0)
        polyline = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        # Chords are marginally shorter than the arcs they approximate.
        assert abs(polyline - circuit.length_m) < circuit.length_m * 0.005, name
        assert abs(float(arc[-1]) - circuit.length_m) < circuit.length_m * 0.005, name


def test_reference_line_centreline_projects_onto_itself():
    for name, circuit in CIRCUITS.items():
        line = ReferenceLine(circuit, spacing=2.0)
        for index in np.linspace(0, len(line.points) - 1, 40).astype(int):
            _s, lateral = line.project(line.points[index])
            assert abs(lateral) < 1e-6, f"{name} at point {index}"


def test_reference_line_start_is_zero():
    for circuit in CIRCUITS.values():
        line = ReferenceLine(circuit)
        s, lateral = line.project(line.points[0])
        assert s == pytest.approx(0.0, abs=1e-6)
        assert lateral == pytest.approx(0.0, abs=1e-6)


def test_lateral_sign_is_positive_to_the_left():
    line = ReferenceLine(CIRCUITS["monaco"])
    position, tangent = line.sample(0.0)
    left_normal = np.array([-tangent[1], tangent[0]])
    offset = position + left_normal * 4.0
    _s, lateral = line.project(offset)
    assert lateral == pytest.approx(4.0, abs=0.05)
    assert lateral > 0.0


def test_sample_and_project_agree_on_lateral():
    line = ReferenceLine(CIRCUITS["silverstone"])
    for arc in (100.0, 500.0, 1500.0, 3000.0):
        position, tangent = line.sample(arc)
        _s, lateral = line.project(position)
        assert abs(lateral) < 1e-6


def test_reference_line_rejects_nonpositive_spacing():
    with pytest.raises(ValueError):
        ReferenceLine(CIRCUITS["suzuka"], spacing=0.0)
