Similarity Metrics
==================

Similarity is the primary query mechanism in HDC: given a noisy or
transformed hypervector, similarity to each item in a codebook identifies the
nearest match. All metrics in PyHDC return values in **[-1, 1]** (1 =
identical, 0 = orthogonal, -1 = maximally dissimilar).

All similarity functions are in ``pyhdc.components.similarity``.

CosineSimilarity
-----------------

**Used by**: MAP_C, MAP_I, MAP_I_Bits, MAP_B, HRR family, FHRR, VTB, MBAT

.. math::

   \text{cos}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\|_2 \|\mathbf{b}\|_2}

Output range: [-1, 1]

Cosine similarity is the dot product of unit vectors. It measures the angle
between two vectors, independent of their magnitudes. This makes it
appropriate for both normalised (HRR) and unnormalised (MAP) vectors.

For two random unit vectors in :math:`\mathbb{R}^D`, the expected cosine
similarity is 0 and the standard deviation is :math:`1/\sqrt{D}`.

HammingDistance
----------------

**Used by**: BSC

Normalised and remapped Hamming distance:

.. math::

   \text{hamming}(\mathbf{a}, \mathbf{b}) = 1 - 2 \cdot \frac{\text{popcount}(\mathbf{a} \oplus \mathbf{b})}{D}

Output range: [-1, 1]

The formula maps:

* 0 bit flips (identical vectors) -> 1.0
* D/2 bit flips (random/orthogonal) -> 0.0
* D bit flips (all-different) -> -1.0

This is consistent with the [-1, 1] convention used by all other metrics.

.. note::

   **v1.0.x breaking change**: In v1.0.x, ``HammingDistance`` returned
   ``popcount(a XOR b) / D``, in [0, 1] with 0 = identical and 1 = all-different.
   Code that compared against thresholds in [0, 1] must be updated, or use
   ``similarity_remap=remap_to_unit`` on the encoding constructor to restore
   the [0, 1] output.

Overlap
--------

**Used by**: BSDC family

Normalised set intersection, remapped to [-1, 1]:

.. math::

   \text{overlap\_raw}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\min(\|\mathbf{a}\|_1,\, \|\mathbf{b}\|_1)}

.. math::

   \text{overlap}(\mathbf{a}, \mathbf{b}) = 2 \cdot \text{overlap\_raw} - 1

Output range: [-1, 1]

For sparse binary vectors, the dot product counts the number of positions
where both vectors have a 1. Dividing by the smaller :math:`\ell_1` norm
(i.e., the smaller number of 1s) gives a Jaccard-like coefficient:
0 = no overlap, 1 = the smaller vector is a subset of the larger.

.. note::

   Same v1.0.x breaking change as HammingDistance.

AngleDistance
--------------

**Used by**: FHRR

For angle-valued vectors, similarity is the cosine of the mean angular
difference:

.. math::

   \text{angle\_dist}(\mathbf{a}, \mathbf{b}) = \frac{1}{D} \sum_{i=1}^D \cos(a_i - b_i)

Output range: [-1, 1]

This is appropriate because FHRR binding uses modular angle arithmetic:
two vectors are "similar" when their angles are close element-wise (small
absolute angular difference per dimension).

remap_to_unit
--------------

A utility function that maps any [-1, 1] similarity value to [0, 1]:

.. math::

   \text{remap\_to\_unit}(s) = \frac{s + 1}{2}

This maps: -1 -> 0, 0 -> 0.5 (orthogonal), +1 -> 1.

It works on scalars, NumPy arrays, and PyTorch tensors. Use it as the
``similarity_remap=`` argument on any encoding to apply it automatically.

Batched calling conventions
-----------------------------

Batches are dimension-first: a batch of N hypervectors has shape ``(D, N)``,
where each column ``batch[:, i]`` is one hypervector. Similarity operates
column-wise over axis 0. The supported input modes:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Input shape
     - Output shape
     - Semantics
   * - ``(D,)`` and ``(D,)``
     - scalar
     - Single pair
   * - ``(D, N)`` and ``(D, N)``
     - ``(N,)``
     - Per-column pairs: ``result[i] = sim(A[:, i], B[:, i])``
   * - ``(D,)`` and ``(D, N)``
     - ``(N,)``
     - One vector vs. each column: ``result[i] = sim(v, B[:, i])``
   * - ``(D, N)`` (single arg)
     - ``(N-1,)``
     - Column 0 vs. columns 1...N-1: ``result[i] = sim(batch[:, 0], batch[:, i+1])``
   * - ``[a, b]`` and ``[c, d]`` (equal-length lists)
     - list of scalars
     - Pairwise: ``[sim(a, c), sim(b, d)]``

Cross-similarity mode
---------------------

``similarity(A, B, mode="cross")`` with ``A`` of shape ``(D, P)`` and ``B`` of
shape ``(D, M)`` returns the full ``(P, M)`` cross-similarity matrix: ``result[i, j]`` is the
similarity of column ``A[:, i]`` against column ``B[:, j]``. The result is
backed by a single matmul, with no ``(D, P, M)`` intermediate.

Cross mode requires two batch operands. Passing only one operand raises
``ValueError`` at every entry point. The strict input guards live on
``Encoding.similarity``: passing a Python-list operand or an ``axis`` argument also
raises ``ValueError``. The ``similarity_remap`` callback, when set, is applied to
the ``(P, M)`` result.

The supported metrics are :func:`~pyhdc.similarity` over Cosine, Hamming,
Overlap, and Angle. A metric outside that set raises ``NotImplementedError``.
Every PyHDC provided encoding maps to one of these four, so a built-in encoding 
will always take the matmul path.

Per-metric behavior in cross mode:

* **Cosine** normalizes both operands, then takes the matmul. A zero-norm column
  is guarded so it scores 0 rather than ``nan``.
* **Hamming** casts the operands to ``float64`` for the matmul.
* **Overlap** keeps an asymmetric denominator: each column ``j`` is divided by
  the nonzero count of ``B[:, j]``. Pass the bundled vectors as ``A`` and the
  reference codebook as ``B``.
* **Angle** uses the identity ``cos(x - y) = cos x cos y + sin x sin y``,
  computed as two matmuls.

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_I(dimension=10_000)

   A = enc.generate(size=(10_000, 5))   # (D, P) = (D, 5)
   B = enc.generate(size=(10_000, 8))   # (D, M) = (D, 8)

   scores = pyhdc.similarity(A, B, mode="cross")   # (P, M) = (5, 8)
   print(scores.shape)                             # (5, 8)

Axis-aware reduction and trailing-axis broadcasting
----------------------------------------------------

Every metric reduces over **axis 0**, the hypervector dimension :math:`D`.
The result shape is the broadcast of the two operands' trailing axes (axes 1
and higher). The dimension axis disappears in the reduction. This is what
lets a higher-rank batch line up against a lower-rank one. A ``(D, N)`` input
compared against a ``(D, N, M)`` input pads the smaller operand to
``(D, N, 1)`` and broadcasts over the last axis, yielding an ``(N, M)`` score
array. The ``axis=`` keyword on :func:`~pyhdc.similarity` is keyword-only and,
for a single batched input, selects which batch axis splits index 0 from the
rest, the reduction itself stays on axis 0.

A Python ``float`` comes back only when both operands are 1D (``(D,)`` against
``(D,)``). Every other case returns a NumPy array or PyTorch tensor whose
shape is the broadcast of the non-dimension axes. The ``similarity_remap``
callback, when set, is applied to that result.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - A shape
     - B shape
     - Result
   * - ``(D,)``
     - ``(D,)``
     - Python ``float`` (scalar)
   * - ``(D,)``
     - ``(D, N)``
     - ``(N,)``
   * - ``(D, N)``
     - ``(D,)``
     - ``(N,)``
   * - ``(D, N)``
     - ``(D, N)``
     - ``(N,)``
   * - ``(D,)``
     - ``(D, N, M)``
     - ``(N, M)``
   * - ``(D, N, M)``
     - ``(D,)``
     - ``(N, M)``
   * - ``(D, N)``
     - ``(D, N, M)``
     - ``(N, M)`` (A padded to ``(D, N, 1)``, broadcast over M)
   * - ``(D, N, M)``
     - ``(D, N)``
     - ``(N, M)``
   * - ``(D, N, M)``
     - ``(D, N, M)``
     - ``(N, M)``
   * - ``(D, 1, M)``
     - ``(D, N, M)``
     - ``(N, M)`` (broadcast over axis 1)

Single-input ``similarity`` on a ``(D, N, M, ...)`` batch (``ndim >= 3``)
requires an explicit ``axis``. Passing a 1D single input or omitting the axis
on a 3D+ single input raises ``ValueError``. The chosen axis must resolve to
exactly one batch axis, and axis 0 is never reducible.

.. code-block:: python

   import numpy as np
   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)

   query = enc.generate(size=(10_000, 4))        # (D, N) = (D, 4)
   codebook = enc.generate(size=(10_000, 4, 8))  # (D, N, M) = (D, 4, 8)

   scores = enc.similarity(query, codebook)      # (N, M) = (4, 8)
   print(scores.shape)                           # (4, 8)

Choosing the right metric
--------------------------

The encoding automatically selects the appropriate metric, you do not need
to call these functions directly. The mapping is:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Encoding
     - Metric
   * - MAP_C, MAP_I, MAP_I_Bits, MAP_B
     - CosineSimilarity
   * - HRR, HRR_NoNorm, HRR_ConstNorm
     - CosineSimilarity
   * - FHRR
     - AngleDistance
   * - VTB, MBAT
     - CosineSimilarity
   * - BSC
     - HammingDistance
   * - BSDC_CDT, BSDC_S, BSDC_SEG, BSDC_THIN
     - Overlap
