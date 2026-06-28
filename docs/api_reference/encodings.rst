Encoding Classes
================

All 15 encoding classes inherit from :class:`~pyhdc.Encoding` and share the
same constructor signature and method set. See :doc:`encoding_base` for the
common interface.

This page documents the encoding-specific parameters and characteristics.

.. currentmodule:: pyhdc

MAP family
----------

.. autoclass:: MAP_C
   :show-inheritance:

**Element type**: float in [-1, 1], default dtype ``float32``

**Extra parameters** (in addition to :class:`Encoding` base parameters):

* ``random_choice_range``: ``float`` or ``None`` (rho): controls band
  randomization during bundling. Coordinates whose pre-aggregate ``|sum|``
  falls within the band are replaced by independent ``Uniform[-1, 1]`` draws.
  Reduces systematic bias. Default: ``None`` (no band randomization).

**Binding**: ElementMultiplication (element-wise product, self-inverse)

**Bundling**: ElementAdditionCut (element-wise sum then clip to [-1, 1])

**Similarity**: CosineSimilarity → [-1, 1]

**Unbind**: Yes

**Generator output type required**: ``"floats"``

----

.. autoclass:: MAP_I
   :show-inheritance:

**Element type**: int {-1, +1}, default dtype ``int32``

**Extra parameters**: ``random_choice_range`` (same as MAP_C)

**Binding**: ElementMultiplication

**Bundling**: ElementAdditionCut

**Similarity**: CosineSimilarity

**Unbind**: Yes

**Generator output type required**: ``"bits"``

----

.. autoclass:: MAP_I_Bits
   :show-inheritance:

**Element type**: bipolar ``{-1, +1}``, saturating to a signed integer after
bundling. Storage dtype auto-widens to the smallest signed type that holds the
configured width (``int8`` / ``int16`` / ``int32`` / ``int64``), default ``int32``.

**Extra parameters**:

* ``mask``: ``int``, default ``(2**32) - 1``. Must have the form ``2**k - 1``
  (contiguous low bits). Selects the signed saturation width ``k``. A mask that is
  not ``2**k - 1`` raises ``ValueError`` (use ``bit_width`` instead). Example:
  ``mask=0xFF`` (= ``2**8 - 1``) gives 8-bit saturation in ``int8``.
* ``bit_width``: ``int`` or ``None``, default ``None``. Explicit signed bit width
  ``k`` (1-64), overrides ``mask`` when set.

**Binding**: ElementMultiplication. Bipolar operands stay bipolar, but binding a
*bundled* (saturated, non-bipolar) vector at a narrow width can overflow the
storage dtype, since the product is not re-clipped. Bind before bundling, or use a
wider width, if you need to bind sums.

**Bundling**: ElementAdditionBits (element-wise sum in a wide accumulator, then a
single saturating clip to the signed ``k``-bit range)

**Similarity**: CosineSimilarity

**Unbind**: Yes

**Generator output type required**: ``"words"``

----

.. autoclass:: MAP_B
   :show-inheritance:

**Element type**: bipolar {-1, +1}, default dtype ``int8``

**Extra parameters**: ``random_choice_range``

**Binding**: ElementMultiplication

**Bundling**: ElementAdditionBipolarThreshold (element-wise sum, then sign to {-1, +1})

**Similarity**: CosineSimilarity

**Unbind**: Yes

**Generator output type required**: ``"bits"``

----

HRR family
----------

.. autoclass:: HRR
   :show-inheritance:

**Element type**: normal float, default dtype ``float32``

**Extra parameters**: ``random_choice_range``

**Binding**: CircularConvolution (via FFT)

**Bundling**: ElementAdditionNormalized (sum then L2 normalise)

**Similarity**: CosineSimilarity

**Unbind**: Yes (via CircularCorrelation)

**Generator output type required**: ``"floats"``

----

.. autoclass:: HRR_NoNorm
   :show-inheritance:

Like :class:`HRR` but bundling does **not** normalise the result.
Vector magnitude grows with the number of bundled items.

----

.. autoclass:: HRR_ConstNorm
   :show-inheritance:

Like :class:`HRR` but bundling divides by :math:`\sqrt{M}` rather than the
L2 norm. Maintains constant expected magnitude.

----

.. autoclass:: FHRR
   :show-inheritance:

**Element type**: angles in [-π, π), default dtype ``float32``

**Extra parameters**: ``random_choice_range``

**Binding**: ElementAngleAddition (modular angle addition)

**Bundling**: AnglesOfElementAddition (phasor resultant angle)

**Similarity**: AngleDistance → [-1, 1]

**Unbind**: Yes (ElementAngleSubtraction)

**Generator output type required**: ``"floats"``

----

Matrix family
-------------

.. autoclass:: VTB
   :show-inheritance:

**Element type**: normal float, default dtype ``float32``

**Binding**: VectorDerivedTransformation (matrix derived from the key vector)

**Bundling**: ElementAdditionNormalized

**Similarity**: CosineSimilarity

**Unbind**: Yes (transpose of the transformation matrix)

**Generator output type required**: ``"floats"``

----

.. autoclass:: MBAT
   :show-inheritance:

**Element type**: normal float, default dtype ``float32``

**Binding**: MatrixMultiplication (random matrix)

**Bundling**: ElementAdditionNormalized

**Similarity**: CosineSimilarity

**Unbind**: Yes; requires ``get_metadata()["matrices"]`` from the bound result.

.. note::

   MBAT stores the random binding matrices in the result's metadata. You must
   preserve ``bound_hv.get_metadata()`` to perform unbinding later.

**Generator output type required**: ``"floats"``

----

Binary family
-------------

.. autoclass:: BSC
   :show-inheritance:

**Element type**: binary {0, 1}, default dtype ``int8``

**Extra parameters**: ``random_choice_range``

**Binding**: ExclusiveOr (XOR; exactly self-inverse)

**Bundling**: ElementAdditionBinaryThreshold (majority vote)

**Similarity**: HammingDistance → [-1, 1]

**Unbind**: Yes (exact, not approximate)

**Generator output type required**: ``"bits"``

----

Sparse binary family (BSDC)
----------------------------

All BSDC variants use sparse binary vectors (density ≈ 1–5%), bitwise OR for
bundling, and Overlap similarity.

.. autoclass:: BSDC_CDT
   :show-inheritance:

**Binding**: AdditiveContextDependentThinning

**Bundling**: Disjunction (OR; density grows without thinning)

**Similarity**: Overlap → [-1, 1]

**Unbind**: **No**; ``unbind()`` raises ``NotImplementedError``

----

.. autoclass:: BSDC_S
   :show-inheritance:

**Binding**: Shifting (circular shift by one position per bind step)

**Bundling**: Disjunction (OR; density grows)

**Similarity**: Overlap → [-1, 1]

**Unbind**: Yes (inverse circular shift)

----

.. autoclass:: BSDC_SEG
   :show-inheritance:

Like :class:`BSDC_S` but the circular shift is applied independently to each
segment of the vector.

----

.. autoclass:: BSDC_THIN
   :show-inheritance:

*Added in v1.1.0*

**Binding**: Shifting (same as BSDC_S)

**Bundling**: DisjunctionThinned (OR then random thinning to maintain density)

**Similarity**: Overlap → [-1, 1]

**Unbind**: Yes (inverse circular shift)

The recommended sparse binary default when you need stable density over many
bundle operations.
