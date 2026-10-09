"""Racing circuits for MetaDrive, expressed as block sequences.

MetaDrive builds road networks by chaining *blocks*. A racing circuit is exactly
the block vocabulary MetaDrive already speaks:

``Straight(length)``
    A run of road ``length`` metres long.

``Curve(length, radius, angle, dir)``
    ``create_bend_straight`` builds this as an **arc** followed by a **straight**:
    the arc has the given ``radius`` and sweeps ``angle`` degrees, then the block
    continues straight for ``length`` metres. The arc's length is *derived*
    (``radius * radians(angle)``) -- ``length`` is the run-off, not the arc.
    ``dir=1`` is clockwise (a right-hander), ``dir=0`` counter-clockwise (left).

Because a circuit is a closed loop, every block sequence must satisfy two
geometric conditions: the total turn must be a full revolution and the path must
return to where it started. :func:`validate_circuit` checks exactly that, and
:mod:`tests.test_circuits` asserts it for every circuit shipped here -- a mistyped
corner radius breaks the map *silently* otherwise, and a track whose end does not
meet its start cannot be lapped.

Fidelity limits, deliberately accepted and documented rather than hidden:

* **Flat.** MetaDrive blocks are 2D, so Suzuka's ~40 m of elevation change and
  Silverstone's gradients are not represented.
* **Circular corners.** Real corners vary in radius and compound. Curves here are
  constant-radius arcs, so an "S" is written as two opposite-direction arcs.
* **No crossover.** Suzuka's bridge over the back straight is a grade-separated
  crossing that a flat non-intersecting block chain cannot express.
* **No kerbs.** Roads are bounded by guardrails, not rideable kerbing.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

# Directions, matching MetaDrive's `Parameter.dir` convention.
RIGHT = 1  # clockwise
LEFT = 0  # counter-clockwise

BlockKind = Literal["straight", "curve"]


@dataclass(frozen=True)
class Block:
    """One segment of a circuit.

    Args:
        kind: ``"straight"`` or ``"curve"``.
        length: Metres. For a curve this is the *straight run-off after the arc*,
            because the arc length is derived from ``radius`` and ``angle``.
        radius: Arc radius in metres. Curve only.
        angle: Sweep in degrees. Curve only.
        direction: :data:`RIGHT` or :data:`LEFT`. Curve only.
    """

    kind: BlockKind
    length: float
    radius: float = 0.0
    angle: float = 0.0
    direction: int = RIGHT

    def __post_init__(self) -> None:
        if self.kind not in ("straight", "curve"):
            raise ValueError(f"unknown block kind {self.kind!r}")
        if self.length < 0:
            raise ValueError(f"length must be >= 0, got {self.length}")
        if self.kind == "curve":
            if self.radius <= 0:
                raise ValueError(f"curve radius must be > 0, got {self.radius}")
            if self.direction not in (LEFT, RIGHT):
                raise ValueError(f"direction must be LEFT(0) or RIGHT(1), got {self.direction}")

    @property
    def signed_angle(self) -> float:
        """Sweep in radians, positive for a left turn (CCW), negative for right."""
        if self.kind == "straight":
            return 0.0
        return math.radians(self.angle) * (1.0 if self.direction == LEFT else -1.0)

    @property
    def arc_length(self) -> float:
        """Length of the curved part, derived from radius and sweep."""
        if self.kind == "straight":
            return 0.0
        return self.radius * abs(self.signed_angle)

    @property
    def travel_length(self) -> float:
        """Total distance travelled through this block."""
        return self.arc_length + self.length


def straight(length: float) -> Block:
    """A straight of ``length`` metres."""
    return Block("straight", length)


def right(radius: float, angle: float, run_off: float = 0.0) -> Block:
    """A right-hand arc (clockwise) of ``radius`` sweeping ``angle`` degrees."""
    return Block("curve", run_off, radius, angle, RIGHT)


def left(radius: float, angle: float, run_off: float = 0.0) -> Block:
    """A left-hand arc (counter-clockwise) of ``radius`` sweeping ``angle`` degrees."""
    return Block("curve", run_off, radius, angle, LEFT)


@dataclass(frozen=True)
class Circuit:
    """A named, closed racing circuit.

    Args:
        name: Human-readable name.
        blocks: The block sequence, starting at the start/finish line.
        lane_width: Width of the racing surface in metres.
        description: Provenance and fidelity notes.
    """

    name: str
    blocks: tuple[Block, ...]
    lane_width: float = 10.0
    description: str = ""

    @property
    def length_m(self) -> float:
        """Total centreline length of one lap, in metres."""
        return sum(b.travel_length for b in self.blocks)

    @property
    def corner_count(self) -> int:
        """Number of corners (curve blocks)."""
        return sum(1 for b in self.blocks if b.kind == "curve")


@dataclass(frozen=True)
class Geometry:
    """Result of walking a block sequence from the origin heading +x."""

    end: tuple[float, float]
    heading: float
    total_turn_deg: float
    length_m: float


def trace(circuit: Circuit) -> Geometry:
    """Walk a circuit from ``(0, 0)`` heading along ``+x`` and report where it ends.

    Each curve is integrated as a true arc: the heading changes by
    ``signed_angle`` and the displacement follows the arc of ``radius``.
    Each straight advances along the current heading.

    Returns:
        The end point, final heading (radians), total signed turn (degrees) and
        total length.
    """
    x, y, heading = 0.0, 0.0, 0.0
    turn = 0.0
    total = 0.0

    for block in circuit.blocks:
        alpha = block.signed_angle
        total += block.travel_length

        if block.kind == "curve" and alpha != 0.0:
            start_heading = heading
            end_heading = heading + alpha
            radius = block.radius
            # Displacement along a circular arc, integrated exactly.
            x += radius * (math.sin(end_heading) - math.sin(start_heading))
            y += radius * (-math.cos(end_heading) + math.cos(start_heading))
            heading = end_heading
            turn += math.degrees(alpha)

        if block.length:
            x += block.length * math.cos(heading)
            y += block.length * math.sin(heading)

    return Geometry(end=(x, y), heading=heading, total_turn_deg=turn, length_m=total)


class CircuitNotClosedError(ValueError):
    """Raised when a block sequence fails to form a closed loop."""


def validate_circuit(
    circuit: Circuit,
    *,
    position_tol: float = 2.0,
    turn_tol: float = 0.5,
) -> Geometry:
    """Assert that a circuit closes on itself.

    Two conditions, both necessary for the track to be lappable:

    1. The accumulated turn must be one full revolution (+/-360 deg), otherwise
       the road network ends up facing a different direction than it started.
    2. The traced path must return to its start within ``position_tol`` metres,
       otherwise the final block's socket does not meet the first block's.

    Args:
        circuit: The circuit to check.
        position_tol: Allowed distance from the start point, in metres.
        turn_tol: Allowed deviation from a full revolution, in degrees.

    Returns:
        The traced :class:`Geometry`, for callers that want the numbers.

    Raises:
        CircuitNotClosedError: If either condition fails.
    """
    if not circuit.blocks:
        raise CircuitNotClosedError(f"{circuit.name}: circuit has no blocks")

    geom = trace(circuit)
    gap = math.dist(geom.end, (0.0, 0.0))
    turn_error = abs(abs(geom.total_turn_deg) - 360.0)

    if turn_error > turn_tol:
        raise CircuitNotClosedError(
            f"{circuit.name}: net turn is {geom.total_turn_deg:.2f} deg, expected "
            f"+/-360 deg (off by {turn_error:.2f} > {turn_tol})"
        )
    if gap > position_tol:
        raise CircuitNotClosedError(
            f"{circuit.name}: path ends {gap:.2f} m from the start at "
            f"({geom.end[0]:.2f}, {geom.end[1]:.2f}) (tolerance {position_tol} m)"
        )
    return geom


# --------------------------------------------------------------------------- #
# Circuits
#
# These are *approximations* of the real circuits, not surveyed centrelines. Each
# keeps the real circuit's character -- corner radii, corner order, relative
# straight lengths -- because that is what determines how a car drives it. Real
# corner geometry is variable-radius and compound; here every corner is one
# constant-radius arc, so compounds are written as pairs of arcs.
# --------------------------------------------------------------------------- #

SUZUKA = Circuit(
    name="suzuka",
    lane_width=12.0,
    description=(
        "Approximation of Suzuka (5.807 km, 18 turns). Flowing, fast, with a 130R "
        "complex and a very tight hairpin. Flat: the 40 m elevation change and the "
        "bridge crossover over the back straight are not represented."
    ),
    blocks=(
        straight(449.4),
        right(130.0, 70.2, 41.8),
        straight(131.8),
        left(150.0, 47.5, 12.0),
        straight(27.2),
        right(90.0, 62.3, 39.8),
        straight(69.8),
        left(85.0, 67.9, 12.0),
        straight(54.3),
        right(160.0, 56.6, 35.9),
        straight(95.9),
        right(45.0, 118.9, 97.6),
        straight(197.6),
        left(28.0, 198.1, 12.0),
        straight(69.2),
        right(190.0, 81.5, 68.6),
        straight(178.6),
        right(85.0, 54.3, 80.2),
        straight(100.2),
        right(42.0, 88.3, 76.5),
        straight(211.5),
        right(135.0, 77.0, 34.2),
        straight(164.2),
        right(40.0, 70.2, 12.0),
        straight(12.0),
        left(40.0, 70.2, 12.0),
        straight(84.2),
        left(50.0, 79.2, 83.3),
        straight(113.3),
        right(48.0, 81.5, 12.0),
        straight(71.8),
        right(120.0, 62.3, 12.0),
        straight(79.4),
    ),
)

SILVERSTONE = Circuit(
    name="silverstone",
    lane_width=12.0,
    description=(
        "Approximation of Silverstone (5.891 km, 18 turns). Fast and flowing with "
        "heavy braking into Abbey, Farm and Copse. Flat: elevation is not modelled."
    ),
    blocks=(
        straight(485.1),
        right(130.0, 79.0, 48.9),
        straight(118.9),
        right(60.0, 70.2, 13.8),
        straight(63.8),
        left(55.0, 43.9, 12.0),
        straight(12.0),
        right(55.0, 43.9, 12.0),
        straight(143.8),
        left(120.0, 61.5, 17.4),
        straight(47.4),
        right(95.0, 52.7, 28.8),
        straight(48.8),
        left(110.0, 43.9, 12.0),
        straight(176.9),
        right(55.0, 114.1, 87.5),
        straight(147.5),
        right(45.0, 61.5, 78.0),
        straight(133.0),
        left(45.0, 61.5, 52.5),
        straight(167.5),
        right(95.0, 70.2, 103.0),
        straight(183.0),
        right(150.0, 79.0, 95.1),
        straight(185.1),
    ),
)

MONACO = Circuit(
    name="monaco",
    lane_width=9.0,
    description=(
        "Approximation of Monaco (3.337 km, 19 turns). Slowest street circuit on "
        "the calendar: hairpins, a tunnel chicane and a punishingly narrow layout."
    ),
    blocks=(
        straight(249.6),
        right(32.0, 133.7, 57.9),
        straight(97.9),
        right(34.0, 97.7, 75.4),
        straight(145.4),
        right(42.0, 72.0, 12.0),
        straight(43.6),
        right(45.0, 87.4, 12.0),
        straight(12.0),
        left(24.0, 56.6, 12.0),
        straight(12.0),
        right(24.0, 56.6, 12.0),
        straight(12.0),
        right(30.0, 113.1, 68.2),
        straight(128.2),
        left(22.0, 174.9, 12.0),
        straight(37.2),
        right(32.0, 97.7, 12.0),
        straight(18.7),
        left(22.0, 180.0, 56.3),
        straight(141.3),
        right(35.0, 92.6, 12.0),
        straight(12.0),
        right(26.0, 102.9, 12.0),
        straight(30.9),
        left(22.0, 61.7, 12.0),
        straight(12.0),
        right(22.0, 61.7, 12.0),
        straight(85.9),
        left(28.0, 82.3, 12.0),
        straight(49.6),
    ),
)


CIRCUITS: dict[str, Circuit] = {
    c.name: c for c in (SUZUKA, SILVERSTONE, MONACO)
}


def get_circuit(name: str) -> Circuit:
    """Look up a circuit by name.

    Args:
        name: Circuit name, case-insensitive.

    Returns:
        The :class:`Circuit`.

    Raises:
        KeyError: If the circuit is unknown.
    """
    try:
        return CIRCUITS[name.lower()]
    except KeyError:
        raise KeyError(f"unknown circuit {name!r}; available: {sorted(CIRCUITS)}") from None


# --------------------------------------------------------------------------- #
# Map construction
# --------------------------------------------------------------------------- #


def build_map_class(circuit: Circuit) -> type:
    """Build a MetaDrive ``PGMap`` subclass that lays out ``circuit``.

    Uses the same plumbing as MetaDrive's own ``RacingMap`` -- a ``FirstPGBlock``
    followed by chained ``Straight``/``Curve`` blocks -- but parameterised by a
    :class:`Circuit` instead of hardcoded geometry. Lanes are bounded by
    guardrails on both sides, since a racing circuit has no lane markings to
    wander across.

    The circuit is validated before the class is returned, so a broken track
    raises here rather than producing a map that silently fails to close.

    Args:
        circuit: The circuit to realise.

    Returns:
        A fresh ``PGMap`` subclass.
    """
    validate_circuit(circuit)

    from metadrive.component.map.pg_map import PGMap
    from metadrive.component.pg_space import Parameter
    from metadrive.component.pgblock.curve import Curve
    from metadrive.component.pgblock.first_block import FirstPGBlock
    from metadrive.component.pgblock.straight import Straight
    from metadrive.constants import PGLineType

    def _generate(self) -> None:
        parent_node_path = self.engine.worldNP
        physics_world = self.engine.physics_world
        if len(self.road_network.graph) != 0:
            raise RuntimeError("map is not empty; create a new map to read config")

        lane_num = self.config["lane_num"]
        lane_width = self.config.get("lane_width", circuit.lane_width)

        previous = FirstPGBlock(
            self.road_network,
            lane_width=lane_width,
            lane_num=lane_num,
            render_root_np=parent_node_path,
            physics_world=physics_world,
            remove_negative_lanes=True,
            side_lane_line_type=PGLineType.GUARDRAIL,
            center_line_type=PGLineType.GUARDRAIL,
        )
        self.blocks.append(previous)

        for index, block in enumerate(circuit.blocks, start=1):
            block_cls = Straight if block.kind == "straight" else Curve
            current = block_cls(
                index,
                previous.get_socket(0),
                self.road_network,
                lane_num,
                remove_negative_lanes=True,
                side_lane_line_type=PGLineType.GUARDRAIL,
                center_line_type=PGLineType.GUARDRAIL,
            )
            if block.kind == "straight":
                params = {Parameter.length: block.length}
            else:
                params = {
                    Parameter.length: block.length,
                    Parameter.radius: block.radius,
                    Parameter.angle: block.angle,
                    Parameter.dir: block.direction,
                }
            current.construct_from_config(params, parent_node_path, physics_world)
            self.blocks.append(current)
            previous = current

    class CircuitMap(PGMap):
        """MetaDrive map realising a single :class:`Circuit`."""

        def _generate(self) -> None:
            _generate(self)

    CircuitMap.__name__ = f"CircuitMap_{circuit.name}"
    CircuitMap.__qualname__ = CircuitMap.__name__
    CircuitMap.__doc__ = f"Generated racing map for the {circuit.name} circuit."
    return CircuitMap


def map_config(circuit: Circuit) -> dict[str, Any]:
    """``map_config`` dict for MetaDrive describing ``circuit``.

    Single lane down the middle with guardrails: a race track has no lane
    markings for the car to straddle.

    Args:
        circuit: The circuit to realise.

    Returns:
        A config dict for use as ``config["map_config"]``.
    """
    return {"lane_num": 1, "lane_width": circuit.lane_width}


def describe(circuit: Circuit) -> Iterable[str]:
    """Human-readable summary lines for a circuit."""
    yield f"{circuit.name}: {circuit.length_m:.0f} m, {circuit.corner_count} corners"
    yield f"  lane width {circuit.lane_width:.1f} m"
    if circuit.description:
        yield f"  {circuit.description}"
