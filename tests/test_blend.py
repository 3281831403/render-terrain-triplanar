"""Contract tests for :mod:`triplanar.blend`.

In the initial repository every stubbed function raises
``NotImplementedError``, so all fifteen tests fail.  The tests are the
specification: do not weaken or edit them while implementing the stubs.
"""

import math

import pytest

from triplanar.blend import (
    TriplanarError,
    axis_weights,
    blend_normals,
    blend_values,
    fix_seams,
    project_uv,
    transition_band,
    triplanar_batch,
    validate,
)


TOL = 1e-6


def test_axis_weights_use_unit_normal_not_absolute_components():
    """Defect 1: a long normal must not add extra brightness on a steep slope."""
    long_normal = axis_weights((3.0, 4.0, 0.0), 1.0)
    unit_normal = axis_weights((0.6, 0.8, 0.0), 1.0)

    assert long_normal == pytest.approx((0.6, 0.8, 0.0), abs=TOL)
    assert long_normal == pytest.approx(unit_normal, abs=TOL)
    assert math.isclose(sum(long_normal), 1.0, abs_tol=TOL)


def test_y_axis_projection_keeps_u_v_order():
    """Defect 2: the Y projection must use (x, z), not the mirrored (z, x)."""
    pos = (1.5, -2.0, 3.25)

    assert project_uv(pos, "y", 2.0) == pytest.approx((3.0, 6.5), abs=TOL)
    assert project_uv(pos, "x", 2.0) == pytest.approx((6.5, -4.0), abs=TOL)
    assert project_uv(pos, "z", 2.0) == pytest.approx((3.0, -4.0), abs=TOL)


def test_blend_values_normalizes_unnormalized_weights():
    """Defect 3: three weights of 0.5 must average colors, not sum to 1.5."""
    values = (
        (0.2, 0.4, 0.6),
        (0.4, 0.6, 0.2),
        (0.6, 0.2, 0.4),
    )
    weights = (0.5, 0.5, 0.5)

    result = blend_values(values, weights)

    assert result == pytest.approx((0.4, 0.4, 0.4), abs=TOL)
    assert all(0.0 <= channel <= 1.0 for channel in result)


def test_blend_normals_returns_unit_length():
    """Defect 4: blending two axis normals must not leave a 0.707-long vector."""
    blended = blend_normals(
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
        (0.5, 0.5),
    )
    length = math.sqrt(sum(component * component for component in blended))

    assert math.isclose(length, 1.0, abs_tol=TOL)
    assert blended == pytest.approx(
        (math.sqrt(0.5), math.sqrt(0.5), 0.0), abs=TOL
    )
    assert all(component >= -TOL for component in blended)


def test_transition_band_shrinks_on_steep_slope():
    """Defect 5: the band must shrink on a dominant axis, not stay fixed."""
    flat = transition_band((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0), 0.75)
    steep = transition_band((0.94, 0.03, 0.03), 0.75)

    assert math.isclose(flat, 0.5, abs_tol=TOL)
    assert math.isclose(steep, 0.045, abs_tol=TOL)
    assert 0.0 <= steep < flat <= 1.0


def test_fix_seams_unwraps_neighbor_uv():
    """Defect 6: a UV just across the seam must be shifted to the near copy."""
    uv_a = (0.98, 0.5)
    uv_b = (0.02, 0.5)

    kept_a, fixed_b = fix_seams(uv_a, uv_b, 1.0)

    assert kept_a == pytest.approx(uv_a, abs=TOL)
    assert fixed_b == pytest.approx((1.02, 0.5), abs=TOL)
    assert abs(fixed_b[0] - kept_a[0]) <= 0.5 + TOL

    _, unchanged_b = fix_seams(uv_a, uv_b, 0.0)
    assert unchanged_b == pytest.approx(uv_b, abs=TOL)


def test_validate_and_axis_weights_reject_zero_normal():
    """Defect 7: a zero normal must be reported and must not yield NaN weights."""
    problems = validate((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))

    assert problems == ["normal must be non-zero"]
    with pytest.raises(TriplanarError):
        axis_weights((0.0, 0.0, 0.0), 1.0)


def test_validate_rejects_negative_weights():
    """Defect 8: negative weights must be rejected before they invert a color."""
    problems = validate((0.0, 1.0, 0.0), (-0.25, 1.25, 0.0))

    assert problems == ["weights must be non-negative"]
    with pytest.raises(TriplanarError):
        blend_values(
            ((0.2, 0.4, 0.6), (0.6, 0.4, 0.2), (0.1, 0.1, 0.1)),
            (-0.25, 1.25, 0.0),
        )


def test_batch_does_not_share_weight_buffer_between_samples():
    """Defect 9: each sample must keep the weights of its own normal."""
    samples = [
        {"pos": (0.0, 0.0, 0.0), "normal": (0.0, 1.0, 0.0)},
        {"pos": (1.5, 0.25, -2.0), "normal": (1.0, 0.0, 0.0)},
        {"pos": (-1.0, 2.0, 3.0), "normal": (0.0, 0.0, 1.0)},
    ]
    config = {"sharpness": 1.0, "scale": 2.0, "threshold": 0.75}

    result = triplanar_batch(samples, config)
    entries = result["samples"]

    assert entries[0]["weights"] == pytest.approx((0.0, 1.0, 0.0), abs=TOL)
    assert entries[1]["weights"] == pytest.approx((1.0, 0.0, 0.0), abs=TOL)
    assert entries[2]["weights"] == pytest.approx((0.0, 0.0, 1.0), abs=TOL)
    assert entries[1]["uvs"]["y"] == pytest.approx((3.0, -4.0), abs=TOL)


def test_axis_weights_rejects_negative_sharpness():
    """Defect 10: a non-positive sharpness must raise instead of dividing by zero."""
    for bad_sharpness in (-1.0, -0.001, -5.0):
        with pytest.raises(TriplanarError):
            axis_weights((1.0, 2.0, 3.0), bad_sharpness)


def test_up_normal_puts_all_weight_on_y_axis():
    """Combination: an upward normal has Y weight exactly 1."""
    weights = axis_weights((0.0, 2.0, 0.0), 3.0)

    assert weights == pytest.approx((0.0, 1.0, 0.0), abs=TOL)
    assert math.isclose(sum(weights), 1.0, abs_tol=TOL)


def test_axis_weights_sum_to_one():
    """Combination: every normalized weight triple sums to 1 within 1e-6."""
    for normal, sharpness in (
        ((1.0, 2.0, 3.0), 1.5),
        ((0.3, -0.4, 0.5), 2.0),
        ((2.0, 0.0, 0.0), 0.5),
    ):
        weights = axis_weights(normal, sharpness)

        assert math.isclose(sum(weights), 1.0, abs_tol=TOL)
        assert all(0.0 <= weight <= 1.0 for weight in weights)


def test_axis_weights_rejects_zero_sharpness():
    """Combination: sharpness 0 must raise TriplanarError."""
    with pytest.raises(TriplanarError):
        axis_weights((1.0, 2.0, 3.0), 0.0)


def test_batch_with_empty_samples_returns_empty_result():
    """Combination: an empty batch is empty, not an error."""
    result = triplanar_batch(
        [],
        {"sharpness": 2.0, "scale": 4.0, "threshold": 0.5},
    )

    assert result == {"count": 0, "samples": []}


def test_batch_success_preserves_sample_order():
    """Combination: a valid batch keeps every input sample in its original order."""
    samples = [
        {"pos": (3.0, 1.0, 4.0), "normal": (0.6, 0.8, 0.0)},
        {"pos": (-2.0, 5.0, 1.0), "normal": (0.0, 0.0, 1.0)},
        {"pos": (7.0, -1.0, 2.0), "normal": (1.0, 0.0, 0.0)},
    ]
    config = {"sharpness": 2.0, "scale": 1.0, "threshold": 0.5}

    result = triplanar_batch(samples, config)

    assert result["count"] == 3
    assert [entry["pos"] for entry in result["samples"]] == [
        sample["pos"] for sample in samples
    ]
    assert all(
        {"pos", "weights", "uvs", "band"} <= set(entry)
        for entry in result["samples"]
    )
