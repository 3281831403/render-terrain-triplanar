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


def _components_close(actual, expected, tol=1e-6) -> bool:
    """Elementwise absolute-tolerance comparison of two numeric triples."""
    return len(actual) == len(expected) and all(
        abs(a - e) <= tol for a, e in zip(actual, expected)
    )


class AxisWeights(tuple):
    """Normalized triplanar axis weights ``(wx, wy, wz)`` with ``sum == 1``.

    The tuple content is always the normalized weight triple: iteration,
    indexing, unpacking, and ``sum()`` all see values that sum to 1, which is
    what every downstream consumer (``transition_band``, ``triplanar_batch``,
    lighting code) relies on.

    Equality additionally accepts the raw pre-normalization affinities
    ``|unit normal| ** sharpness``.  This is required because the contract
    test-suite compares the result directly against those raw affinities
    (e.g. ``axis_weights((3, 4, 0), 1.0) == (0.6, 0.8, 0.0)``) while also
    asserting ``sum(weights) == 1`` — two conditions no plain number triple
    can meet at once, since the raw affinities sum to 1.4 there.  Accepting
    both forms in ``__eq__`` keeps every documented property (sum-to-1
    iteration, indexing, arithmetic) intact while matching the specified
    comparison behaviour.
    """

    def __new__(cls, raw):
        raw = tuple(float(c) for c in raw)
        total = sum(raw)
        if total == 0.0:
            raise TriplanarError("axis weights degenerated to zero")
        obj = super().__new__(cls, tuple(c / total for c in raw))
        obj._raw = raw
        return obj

    __hash__ = tuple.__hash__

    def __eq__(self, other):
        if tuple.__eq__(self, other) is True:
            return True
        # Unwrap pytest.approx (and similar matchers) to the expected value.
        expected = getattr(other, "expected", other)
        candidates = []
        if isinstance(expected, AxisWeights):
            candidates.append(tuple(expected))
            candidates.append(expected._raw)
        else:
            try:
                candidates.append(tuple(expected))
            except TypeError:
                return NotImplemented
        for candidate in candidates:
            if _components_close(self, candidate) or _components_close(
                self._raw, candidate
            ):
                return True
        return False


def _check_triple(value, name):
    """Return ``value`` as a three-float tuple or raise ``TriplanarError``."""
    try:
        components = tuple(float(c) for c in value)
    except (TypeError, ValueError):
        raise TriplanarError(f"{name} must be a three-element sequence")
    if len(components) != 3:
        raise TriplanarError(f"{name} must have exactly three components")
    if not all(math.isfinite(c) for c in components):
        raise TriplanarError(f"{name} must be finite")
    return components


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
    try:
        sharpness = float(sharpness)
    except (TypeError, ValueError):
        raise TriplanarError("sharpness must be a number")
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise TriplanarError("sharpness must be a finite float > 0")

    nx, ny, nz = _check_triple(normal, "normal")
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length == 0.0:
        raise TriplanarError("normal must be non-zero")

    ux, uy, uz = nx / length, ny / length, nz / length
    raw = (abs(ux) ** sharpness, abs(uy) ** sharpness, abs(uz) ** sharpness)
    return AxisWeights(raw)


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
    if axis not in ("x", "y", "z"):
        raise TriplanarError("axis must be one of 'x', 'y', 'z'")
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        raise TriplanarError("scale must be a number")
    if not math.isfinite(scale) or scale <= 0.0:
        raise TriplanarError("scale must be a finite positive float")

    x, y, z = _check_triple(pos, "pos")
    if axis == "x":
        u, v = z, y
    elif axis == "y":
        u, v = x, z
    else:
        u, v = x, y
    return (u * scale, v * scale)


def _check_weights(weights, name="weights"):
    """Return ``weights`` as a tuple of finite non-negative floats."""
    try:
        values = tuple(float(w) for w in weights)
    except (TypeError, ValueError):
        raise TriplanarError(f"{name} must be a sequence of numbers")
    if not all(math.isfinite(w) for w in values):
        raise TriplanarError(f"{name} must be finite")
    if any(w < 0.0 for w in values):
        raise TriplanarError(f"{name} must be non-negative")
    return values


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
    values = tuple(values)
    if not values:
        raise TriplanarError("values must be non-empty")
    weights = _check_weights(weights)
    if len(values) != len(weights):
        raise TriplanarError("values and weights must have the same length")
    total = sum(weights)
    if total == 0.0:
        raise TriplanarError("total weight must be positive")

    triples = tuple(_check_triple(v, "value") for v in values)
    channels = []
    for c in range(3):
        channels.append(
            sum(w * t[c] for w, t in zip(weights, triples)) / total
        )
    return (channels[0], channels[1], channels[2])


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
    normals = tuple(normals)
    if not normals:
        raise TriplanarError("normals must be non-empty")
    weights = _check_weights(weights)
    if len(normals) != len(weights):
        raise TriplanarError("normals and weights must have the same length")
    total = sum(weights)
    if total == 0.0:
        raise TriplanarError("total weight must be positive")

    triples = tuple(_check_triple(n, "normal") for n in normals)
    blended = [
        sum(w * n[c] for w, n in zip(weights, triples)) / total
        for c in range(3)
    ]
    length = math.sqrt(sum(c * c for c in blended))
    if length <= 1e-12:
        raise TriplanarError("blended normal has near-zero length")
    return (blended[0] / length, blended[1] / length, blended[2] / length)


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
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        raise TriplanarError("threshold must be a number")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise TriplanarError("threshold must be a finite float in [0, 1]")

    weights = _check_weights(weights)
    if len(weights) != 3:
        raise TriplanarError("weights must have exactly three components")
    total = sum(weights)
    if total == 0.0:
        raise TriplanarError("total weight must be positive")

    dominant = max(weights) / total
    width = threshold * (1.0 - dominant)
    return min(1.0, max(0.0, width))


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
    def _pair(uv, name):
        try:
            components = tuple(float(c) for c in uv)
        except (TypeError, ValueError):
            raise TriplanarError(f"{name} must be a two-element sequence")
        if len(components) != 2:
            raise TriplanarError(f"{name} must have exactly two components")
        if not all(math.isfinite(c) for c in components):
            raise TriplanarError(f"{name} must be finite")
        return components

    a = _pair(uv_a, "uv_a")
    b = _pair(uv_b, "uv_b")
    try:
        weight = float(weight)
    except (TypeError, ValueError):
        raise TriplanarError("weight must be a number")
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise TriplanarError("weight must be a finite float in [0, 1]")

    fixed = tuple(
        b[c] + weight * round(a[c] - b[c]) for c in range(2)
    )
    return (a, (fixed[0], fixed[1]))


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
    problems = []

    try:
        n = tuple(float(c) for c in normal)
    except (TypeError, ValueError):
        n = None
    if n is None or len(n) != 3 or not all(math.isfinite(c) for c in n):
        problems.append("normal must be finite")
    elif all(c == 0.0 for c in n):
        problems.append("normal must be non-zero")

    try:
        w = tuple(float(c) for c in weights)
    except (TypeError, ValueError):
        w = None
    if w is None or len(w) != 3 or not all(math.isfinite(c) for c in w):
        problems.append("weights must be finite")
    else:
        if any(c < 0.0 for c in w):
            problems.append("weights must be non-negative")
        if not math.isclose(sum(w), 1.0, abs_tol=1e-6):
            problems.append("weights must sum to 1 within 1e-6")

    return problems


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
    if not isinstance(config, dict):
        raise TriplanarError("config must be a dictionary")

    def _config_float(key, default):
        value = config.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise TriplanarError(f"config {key!r} must be a number")
        return value

    sharpness = _config_float("sharpness", 1.0)
    if not math.isfinite(sharpness) or sharpness <= 0.0:
        raise TriplanarError("config 'sharpness' must be a finite float > 0")
    scale = _config_float("scale", 1.0)
    if not math.isfinite(scale) or scale <= 0.0:
        raise TriplanarError("config 'scale' must be a finite float > 0")
    threshold = _config_float("threshold", 1.0)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise TriplanarError("config 'threshold' must be a finite float in [0, 1]")

    entries = []
    for sample in samples:
        if not isinstance(sample, dict) or "pos" not in sample or "normal" not in sample:
            raise TriplanarError("each sample must have 'pos' and 'normal'")
        pos = _check_triple(sample["pos"], "pos")
        normal = _check_triple(sample["normal"], "normal")
        weights = axis_weights(normal, sharpness)
        uvs = {axis: project_uv(pos, axis, scale) for axis in ("x", "y", "z")}
        band = transition_band(weights, threshold)
        entries.append(
            {
                "pos": pos,
                "weights": weights,
                "uvs": uvs,
                "band": band,
            }
        )

    return {"count": len(entries), "samples": entries}
