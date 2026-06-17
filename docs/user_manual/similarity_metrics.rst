Similarity Metrics
==================

Similarity is the primary query mechanism in HDC: given a noisy or
transformed hypervector, similarity to each item in a codebook identifies the
nearest match. All metrics in PyHDC return values in **[-1, 1]** (1 =
identical, 0 = orthogonal, −1 = maximally dissimilar).

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

* 0 bit flips (identical vectors) → 1.0
* D/2 bit flips (random/orthogonal) → 0.0
* D bit flips (all-different) → -1.0

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

This maps: −1 → 0, 0 → 0.5 (orthogonal), +1 → 1.

It works on scalars, NumPy arrays, and PyTorch tensors. Use it as the
``similarity_remap=`` argument on any encoding to apply it automatically.

Batched calling conventions
-----------------------------

All similarity functions support three input modes since v1.1.0:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Input shape
     - Output shape
     - Semantics
   * - ``(D,)`` and ``(D,)``
     - scalar
     - Single pair
   * - ``(N, D)`` and ``(N, D)``
     - ``(N,)``
     - Per-row pairs: ``result[i] = sim(a[i], b[i])``
   * - ``(N+1, D)`` (single arg)
     - ``(N,)``
     - Row 0 vs. rows 1…N: ``result[i] = sim(batch[0], batch[i+1])``

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
