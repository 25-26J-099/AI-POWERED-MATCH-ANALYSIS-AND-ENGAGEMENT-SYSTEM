"""Player rating calculation using weighted formula."""


def compute_rating(
    vaep: float,
    xt: float,
    xg: float,
    pass_accuracy: float,
    progressive_passes: int,
    progressive_carries: int,
    pressures: int,
    recoveries: int,
    tackles: int,
    interceptions: int,
) -> float:
    """Compute player rating on a 0-10 scale.

    Formula:
        rating = 0.35 * VAEP_norm + 0.25 * xT_norm + 0.15 * xG_norm
                + 0.10 * pass_accuracy_norm + 0.10 * progressive_norm
                + 0.05 * defensive_norm

    All components are normalized to [0, 1] before weighting, then
    the final score is scaled to [0, 10].
    """
    # Normalize each component to [0, 1] using reasonable ranges
    # These ranges are calibrated for a single match

    def clamp01(val: float) -> float:
        return max(0.0, min(1.0, val))

    # VAEP: typically ranges from -0.5 to +1.0 per match
    vaep_norm = clamp01((vaep + 0.5) / 1.5)

    # xT: typically ranges from -0.2 to +0.8 per match
    xt_norm = clamp01((xt + 0.2) / 1.0)

    # xG: typically 0 to 1.0 per match
    xg_norm = clamp01(xg / 1.0)

    # Pass accuracy: 0-100%
    pass_acc_norm = clamp01(pass_accuracy / 100.0)

    # Progressive actions: typically 0-15 per match
    progressive_total = progressive_passes + progressive_carries
    progressive_norm = clamp01(progressive_total / 15.0)

    # Defensive actions: typically 0-20 per match
    defensive_total = pressures + recoveries + tackles + interceptions
    defensive_norm = clamp01(defensive_total / 20.0)

    # Weighted sum
    rating = (
        0.35 * vaep_norm
        + 0.25 * xt_norm
        + 0.15 * xg_norm
        + 0.10 * pass_acc_norm
        + 0.10 * progressive_norm
        + 0.05 * defensive_norm
    )

    # Scale to 0-10
    return round(rating * 10, 2)
