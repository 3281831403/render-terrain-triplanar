"""Contract stubs for triplanar terrain texture blending.

All positions and normals in this module are expressed in world space.  A
sample is a dictionary with a world-space position ``"pos"`` and a world-space
surface ``"normal"``; the normal need not be unit length because every helper
normalizes it internally.  Texture values are RGB triples ``(r, g, b)`` with
components in ``[0, 1]``.

The three triplanar axes are named ``"x"``, ``"y"``, and ``"z"``.  A surface
normal is turned into three non-negative axis weights whose sum is 1.  The
``sharpness`` argument controls blend tightness: larger values concentrate the
weight on the axis closest to the normal and therefore narrow the transition
band, while a sharpness of 1 uses the absolute normalized normal components.

Every numeric comparison in the test suite uses an absolute tolerance of
``1e-6``.  The module is deterministic, offline, and imports only the Python
standard library.  Each public function below is intentionally a stub that
raises ``NotImplementedError`` until its contract is implemented.
"""

import math

__all__ = [
    "TriplanarError",
    "axis_weights",
    "blend_normals",
    "blend_values",
    "fix_seams",
    "project_uv",
    "transition_band",
    "triplanar_batch",
    "validate",
]


class TriplanarError(Exception):
    """Raised when a normal, weight, UV, config, or batch violates its contract."""


def axis_weights(normal, sharpness) -> tuple[float, float, float]:
    """Return normalized triplanar axis weights for a world-space normal.

    Contract:
      * ``normal`` is a three-element world-space normal.  Its magnitude is
        irrelevant: ``(3, 4, 0)`` and ``(0.6, 0.8, 0)`` must produce the same
        weights.  Absolute normal components must never be used without first
        normalizing the vector, otherwise a steep slope gets too much total
        weight and appears too bright.
      * ``sharpness`` is the blend sharpness, a finite float strictly greater
        than 0.  Values less than or equal to zero raise ``TriplanarError``
        instead of dividing by zero.
      * Let ``n`` be ``normal`` normalized to unit length.  The raw axis
        affinities are ``(|n.x| ** sharpness, |n.y| ** sharpness,
        |n.z| ** sharpness)`` and are divided by their sum.  For example, an
        upward normal ``(0, 1, 0)`` returns exactly ``(0.0, 1.0, 0.0)``.
      * The result is a tuple of three finite, non-negative floats whose sum is
        1 within an absolute tolerance of 1e-6.
      * A zero-length or non-finite normal raises ``TriplanarError``.
    """
    raise NotImplementedError("axis_weights")


def project_uv(pos, axis, scale) -> tuple[float, float]:
    """Project a world-space position onto one triplanar axis and return UV.

    Contract:
      * ``pos`` is a finite world-space position triple ``(x, y, z)``.
      * ``axis`` is one of the lowercase strings ``"x"``, ``"y"``, or ``"z"``.
        The projection mapping, chosen so that each axis uses the two world
        axes perpendicular to it, is:

        =====  =============
        axis   returned (u,v)
        =====  =============
        ``x``  ``(z, y)``
        ``y``  ``(x, z)``
        ``z``  ``(x, y)``
        =====  =============

      * ``scale`` is a finite positive texture scale.  Both returned UV
        components are multiplied by ``scale``.  In particular, the Y-axis
        projection must return ``(x * scale, z * scale)`` and must not swap the
        pair, which would mirror the texture.
      * The result is a ``(u, v)`` tuple of floats with absolute accuracy 1e-6.
        UVs are not wrapped here; wrapping is the sampler's responsibility.
      * An unknown axis, a non-finite position, or a non-positive scale raises
        ``TriplanarError``.
    """
    raise NotImplementedError("project_uv")


def blend_values(values, weights) -> tuple[float, float, float]:
    """Blend RGB texture values with normalized triplanar weights.

    Contract:
      * ``values`` is a non-empty sequence of RGB triples and ``weights`` is a
        sequence of the same length.  Every weight is finite and non-negative,
        and at least one weight is greater than zero.
      * Each output channel is::

          sum(weight * value_channel) / sum(weights)

        so the weights are normalized internally even when they do not already
        sum to 1.  For example, weights ``(0.5, 0.5, 0.5)`` must behave like
        ``(1/3, 1/3, 1/3)`` and must not push a constant color above its input
        brightness (the raw sum would incorrectly be 1.5).
      * Blending identical RGB triples returns that same triple regardless of
        the positive weights.  The result is a tuple of three floats with an
        absolute tolerance of 1e-6 for callers comparing values.
      * Empty input, mismatched lengths, a negative weight, or a zero total
        weight raises ``TriplanarError``.
    """
    raise NotImplementedError("blend_values")


def blend_normals(normals, weights) -> tuple[float, float, float]:
    """Blend world-space normals and renormalize the result to unit length.

    Contract:
      * ``normals`` is a non-empty sequence of three-element world-space
        normals and ``weights`` contains the same number of finite,
        non-negative values, at least one of which is positive.  Negative
        weights are invalid because they would invert a contribution.
      * The weighted vector sum is divided by the total weight and then
        normalized to unit length.  Returning the unnormalized weighted sum is
        a defect: it shortens the normal and makes lighting too dark.
      * The returned tuple has length 1 within an absolute tolerance of 1e-6.
        Its components lie in ``[-1, 1]``; for inputs whose components are all
        non-negative, the returned components are non-negative as well.
      * If the blended vector has near-zero length (inputs cancel), the result
        is undefined and ``TriplanarError`` is raised instead of emitting NaNs.
    """
    raise NotImplementedError("blend_normals")


def transition_band(weights, threshold) -> float:
    """Return the triplanar transition-band width in the range 0..1.

    Contract:
      * ``weights`` is a three-element sequence of finite, non-negative axis
        weights with a positive total.  The weights are normalized internally.
      * ``threshold`` is a finite blend-width scale in ``[0, 1]``.  A value of 0
        gives a hard edge (width 0); a value of 1 exposes the full geometric
        spread.
      * Let ``dominant`` be the largest normalized weight.  The raw spread is
        ``1 - dominant`` and the returned width is
        ``threshold * (1 - dominant)``.  A flat normal with equal weights
        therefore has a wide band, while a steep slope whose dominant weight is
        close to 1 has a narrow band instead of a fixed-width hard edge.
      * The result is a finite float in ``[0, 1]``, accurate to an absolute
        tolerance of 1e-6.  Invalid weights or a threshold outside ``[0, 1]``
        raise ``TriplanarError``.
    """
    raise NotImplementedError("transition_band")


def fix_seams(uv_a, uv_b, weight) -> tuple[tuple[float, float], tuple[float, float]]:
    """Align two adjacent UVs across a texture seam.

    Contract:
      * ``uv_a`` and ``uv_b`` are finite UV pairs from neighboring projections.
        They may lie on opposite sides of a tile boundary, for example
        ``(0.98, 0.5)`` and ``(0.02, 0.5)``.
      * ``weight`` is a finite blend confidence in ``[0, 1]``.  At 0 the second
        UV is unchanged; at 1 it is shifted by whole tile offsets to the copy
        nearest ``uv_a``.
      * Component-wise, with ``offset = round(uv_a[c] - uv_b[c])``, return::

          (uv_a, (uv_b[c] + weight * offset for each c))

        At full weight the remaining difference in each component is at most
        0.5, so the seam addresses the same texel neighborhood instead of
        leaving an obvious vertical line.  ``uv_a`` is returned unchanged.
      * The result is a pair of two-float tuples, accurate to an absolute
        tolerance of 1e-6.  Invalid UV or weight input raises
        ``TriplanarError``; neither argument is mutated.
    """
    raise NotImplementedError("fix_seams")


def validate(normal, weights) -> list[str]:
    """Return a list of contract problems for a normal and its axis weights.

    Contract:
      * Return an empty list when ``normal`` is a finite non-zero vector and
        ``weights`` contains three finite, non-negative values that sum to 1
        within an absolute tolerance of 1e-6.
      * Otherwise return one or more human-readable messages.  The messages,
        in check order, are:
          - ``"normal must be finite"``
          - ``"normal must be non-zero"``
          - ``"weights must be finite"``
          - ``"weights must be non-negative"``
          - ``"weights must sum to 1 within 1e-6"``
      * A zero normal must be reported rather than silently producing NaN
        weights.  Negative weights must be reported rather than inverting a
        color contribution.  This helper reports problems; it does not raise
        ``TriplanarError`` for them.
      * Neither argument is mutated.
    """
    raise NotImplementedError("validate")


def triplanar_batch(samples, config) -> dict:
    """Evaluate a batch of world-space samples atomically.

    Contract:
      * ``samples`` is a sequence of dictionaries.  Every sample has
        ``"pos"`` (a finite world-space triple) and ``"normal"`` (a finite,
        non-zero world-space normal).  Extra keys are ignored.
      * ``config`` is a dictionary with three optional keys:
        ``"sharpness"`` (finite float > 0, default 1.0), ``"scale"`` (finite
        float > 0, default 1.0), and ``"threshold"`` (finite float in [0, 1],
        default 1.0).  Missing keys use those defaults; invalid values raise
        ``TriplanarError``.
      * On success return::

          {
              "count": len(samples),
              "samples": [
                  {
                      "pos": (x, y, z),
                      "weights": (wx, wy, wz),
                      "uvs": {
                          "x": (u, v),
                          "y": (u, v),
                          "z": (u, v),
                      },
                      "band": float,
                  },
                  ...
              ],
          }

        The inner list preserves input order.  Each entry has its own fresh
        weights tuple; a mutable weight buffer must not be shared between
        samples, or one sample's normal will contaminate later entries.
      * Each sample is computed from its own position and normal using
        :func:`axis_weights`, :func:`project_uv`, and
        :func:`transition_band`.  All numbers are accurate to an absolute
        tolerance of 1e-6.
      * The operation is atomic: if any sample or config value is invalid,
        raise ``TriplanarError`` and return no partial batch.  An empty
        ``samples`` sequence returns ``{"count": 0, "samples": []}``.
      * Inputs are not mutated.
    """
    raise NotImplementedError("triplanar_batch")
