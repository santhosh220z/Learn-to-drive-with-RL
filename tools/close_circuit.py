"""Offline authoring tool: close a circuit's block sequence.

A circuit has to satisfy three geometric conditions at once: total turn must be a
full revolution, the path must return to its start, and every straight must keep a
positive length. Hand-tuning those numbers is tedious and easy to get subtly wrong.

This tool closes a circuit deterministically in two stages:

1. **Balance the turn.** Corner angles are scaled in proportion to their magnitude
   until the net signed turn is a full revolution. Scaling proportionally keeps the
   *relative* severity of each corner, which is what gives a circuit its character.
2. **Close the position.** With the angles now fixed, every straight's heading is
   fixed, so the displacement is a *linear* function of the straight lengths:
   ``sum(straight_length_i * heading_i) == -arc_displacement``. That is a small
   constrained least-squares problem, solved with a lower bound so no straight
   collapses to zero.

It prints the corrected block list. The numbers that come out are pasted back into
``circuits.py`` as static data, so the shipped circuits are plain literals that
``tests/test_circuits.py`` can validate independently.

Usage:
    python tools/close_circuit.py --list
    python tools/close_circuit.py suzuka
    python tools/close_circuit.py suzuka --emit
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drive_rl.circuits import (  # noqa: E402
    CIRCUITS,
    Circuit,
    CircuitNotClosedError,
    trace,
    validate_circuit,
)

MIN_STRAIGHT = 12.0  # metres; shorter than this and the map reads as a kink
STRAIGHT_ANCHOR = 0.02  # how strongly to hold straight lengths near their authored values
MAX_RESIDUAL = 1.0  # metres of closure error tolerated


def balance_turn(circuit: Circuit) -> Circuit:
    """Scale corner angles proportionally until the net turn is one revolution.

    Scaling every angle by the same factor is *exact* -- the signed turns sum
    linearly -- and preserves the relative severity of each corner, which is what
    gives a circuit its character. It only distorts a circuit if the authored data
    is badly unbalanced, so that case is reported instead of hidden.
    """
    current = trace(circuit).total_turn_deg
    if abs(abs(current) - 360.0) < 1e-9:
        return circuit
    if current == 0.0:
        raise CircuitNotClosedError(
            f"{circuit.name}: net turn is 0 deg -- the corner list has no net bias, "
            "so it can never close. Re-author the circuit."
        )

    target = 360.0 if current >= 0 else -360.0
    scale = target / current

    scaled = [b.angle * scale for b in circuit.blocks if b.kind == "curve"]
    wildest = max(scaled, default=0.0)
    if not 0.5 <= scale <= 2.0 or wildest > 300.0:
        print(
            f"[warn] {circuit.name}: turning balance factor {scale:.2f} (largest corner "
            f"{wildest:.0f} deg). The authored corner list is probably unbalanced; "
            f"re-author it rather than relying on this scale.",
            file=sys.stderr,
        )

    new_blocks = tuple(
        block
        if block.kind == "straight"
        else type(block)(
            block.kind, block.length, block.radius, block.angle * scale, block.direction
        )
        for block in circuit.blocks
    )
    return Circuit(
        circuit.name,
        new_blocks,
        circuit.lane_width,
        circuit.description,
    )


def close_position(circuit: Circuit, *, max_rounds: int = 60) -> Circuit:
    """Solve for straight lengths that return the path to its start point.

    With the corner angles now fixed, every straight's heading is fixed, so the
    displacement is a *linear* function of the straight lengths::

        sum(straight_length_i * heading_i) == -arc_displacement

    That is a small constrained least-squares problem. ``STRAIGHT_ANCHOR`` trades
    loop closure against fidelity to the authored proportions: near zero closes
    the loop tightly but lets links between corners shrink to the floor, while a
    large value preserves proportions and leaves a gap the loop cannot tolerate.

    Note the floor is deliberately low. Authored circuits that do not close are
    usually *too long*, so the solve has to shorten straights to succeed -- a
    generous floor removes the slack it needs and it pins every straight against
    the bound with a large residual.
    """
    # Per-block headings and arc displacements, with angles now fixed.
    headings: list[float] = []
    arc_dx = arc_dy = 0.0
    heading = 0.0

    for block in circuit.blocks:
        alpha = block.signed_angle
        if block.kind == "curve" and alpha != 0.0:
            start_h, end_h = heading, heading + alpha
            arc_dx += block.radius * (math.sin(end_h) - math.sin(start_h))
            arc_dy += block.radius * (math.cos(start_h) - math.cos(end_h))
            heading = end_h
        headings.append(heading)

    # Columns of A are the fixed unit headings; we solve A @ lengths = -arc_disp.
    A = np.array([[math.cos(h), math.sin(h)] for h in headings], dtype=float).T
    b = np.array([-arc_dx, -arc_dy], dtype=float)

    target = np.array([blk.length for blk in circuit.blocks], dtype=float)
    weight = STRAIGHT_ANCHOR
    A_aug = np.vstack([A, weight * np.eye(len(target))])
    b_aug = np.concatenate([b, weight * target])

    result = lsq_linear(
        A_aug,
        b_aug,
        bounds=(np.full(len(target), MIN_STRAIGHT), np.full(len(target), 6000.0)),
        method="trf",
        max_iter=max_rounds * 20,
    )

    new_blocks = tuple(
        type(blk)(blk.kind, float(length), blk.radius, blk.angle, blk.direction)
        for blk, length in zip(circuit.blocks, result.x, strict=True)
    )
    closed = Circuit(circuit.name, new_blocks, circuit.lane_width, circuit.description)

    # Measure the residual with `trace`, not with the linear algebra above: trace()
    # is what tests/test_circuits.py asserts against, so validating against it
    # keeps the authoring tool and the test suite from disagreeing.
    gap = math.dist(trace(closed).end, (0.0, 0.0))
    if gap > MAX_RESIDUAL:
        raise CircuitNotClosedError(
            f"{circuit.name}: could not close the loop, residual {gap:.1f} m "
            f"(solver status {result.status})"
        )
    return closed


def close(circuit: Circuit, position_tol: float = 2.0) -> Circuit:
    """Balance the turn, then close the position."""
    return close_position(balance_turn(circuit))


def emit(circuit: Circuit) -> str:
    """Render a circuit as copy-pasteable Python block tuples."""
    lines = [f"# {circuit.name}: {circuit.length_m:.0f} m, {circuit.corner_count} corners"]
    for block in circuit.blocks:
        if block.kind == "straight":
            lines.append(f"straight({block.length:.1f}),")
        elif block.direction == 1:
            lines.append(
                f"right({block.radius:.1f}, {block.angle:.1f}, {block.length:.1f}),"
            )
        else:
            lines.append(
                f"left({block.radius:.1f}, {block.angle:.1f}, {block.length:.1f}),"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Close circuit block sequences.")
    parser.add_argument("name", nargs="?", help="Circuit to close.")
    parser.add_argument("--list", action="store_true", help="List available circuits.")
    parser.add_argument("--emit", action="store_true", help="Print closed block tuples.")
    parser.add_argument("--tolerance", type=float, default=2.0, help="Position tolerance (m).")
    args = parser.parse_args(argv)

    if args.list:
        for name in sorted(CIRCUITS):
            geom = trace(CIRCUITS[name])
            gap = math.dist(geom.end, (0.0, 0.0))
            print(
                f"{name:12s} len={geom.length_m:7.0f}m "
                f"turn={geom.total_turn_deg:8.2f} gap={gap:8.2f}m"
            )
        return 0

    if not args.name:
        parser.error("provide a circuit name or --list")

    original = CIRCUITS[args.name]
    fixed = close(original, args.tolerance)

    before, after = trace(original), trace(fixed)
    print(f"=== {original.name} ===")
    print(f"before: turn={before.total_turn_deg:9.2f} deg  gap={math.dist(before.end, (0, 0)):8.2f} m  len={before.length_m:.0f} m")
    print(f"after : turn={after.total_turn_deg:9.2f} deg  gap={math.dist(after.end, (0, 0)):8.2f} m  len={after.length_m:.0f} m")
    try:
        validate_circuit(fixed, position_tol=args.tolerance)
        print("closes: OK")
    except CircuitNotClosedError as exc:
        print(f"closes: FAILED -> {exc}")
        return 1
    if args.emit:
        print()
        print(emit(fixed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
