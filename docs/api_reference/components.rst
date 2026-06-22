Components Module
=================

.. module:: pyhdc.components

The ``pyhdc.components`` module contains the low-level building blocks used
internally by all encodings. Most users do not need to import these directly;
they are assembled automatically by the encoding classes via ``EncodingSpec``.

Use the components module when:

* Writing a custom encoding subclass
* Applying a specific operation outside an encoding context
* Using ``remap_to_unit`` for post-processing

pyhdc.components.similarity
-----------------------------

.. currentmodule:: pyhdc.components.similarity

.. autofunction:: CosineSimilarity

   Cosine similarity between two vectors or batches.

   :returns: ``float``, ``ndarray``, or ``Tensor`` in [-1, 1].
             Operates column-wise over axis 0 of a ``(D, N)`` batch. See the
             :ref:`batched calling conventions <similarity-batched>`.

.. autofunction:: HammingDistance

   Normalised Hamming distance, mapped to [-1, 1].

   Appropriate for dense binary vectors (BSC encoding).

   :returns: Value in [-1, 1]. (Was [0, 1] in v1.0.x.)

.. autofunction:: Overlap

   Normalised overlap (set intersection), mapped to [-1, 1].

   Appropriate for sparse binary vectors (BSDC family).

   :returns: Value in [-1, 1]. (Was [0, 1] in v1.0.x.)

.. autofunction:: AngleDistance

   Cosine of element-wise angle differences.

   Appropriate for phase/angle encodings (FHRR).

   :returns: Value in [-1, 1].

.. autofunction:: remap_to_unit

   Map a similarity value from [-1, 1] to [0, 1]:
   ``(sim + 1) / 2``.

   :param sim: Scalar, ``ndarray``, or ``Tensor``.
   :returns: Same type as input, values in [0, 1].

pyhdc.components.binding
--------------------------

.. currentmodule:: pyhdc.components.binding

.. autofunction:: ElementMultiplication

   Element-wise product of bipolar vectors. Self-inverse. Used by MAP family.

.. autofunction:: CircularConvolution

   Circular convolution via FFT. Used by HRR family for binding.

.. autofunction:: CircularCorrelation

   Circular correlation via FFT. Used by HRR family for unbinding.

.. autofunction:: ExclusiveOr

   Element-wise XOR. Exactly self-inverse. Used by BSC.

.. autofunction:: Shifting

   Circular shift by one position. Used by BSDC_S, BSDC_THIN for binding.

.. autofunction:: InverseShifting

   Circular shift in the reverse direction. Used for unbinding in BSDC_S.

.. autofunction:: SegmentShifting

   Per-segment circular shift. Used by BSDC_SEG for binding.

.. autofunction:: InverseSegmentShifting

   Per-segment inverse shift. Used by BSDC_SEG for unbinding.

.. autofunction:: AdditiveContextDependentThinning

   CDT binding. Not invertible. Used by BSDC_CDT.

.. autofunction:: VectorDerivedTransformation

   Matrix derived from key vector, applied to value. Used by VTB for binding.

.. autofunction:: TransposeVectorDerivedTransformation

   Transpose of VDT matrix. Used by VTB for unbinding.

.. autofunction:: MatrixMultiplication

   Random matrix multiplication. Used by MBAT for binding. Returns
   ``(result, matrices)``; matrices must be saved for unbinding.

.. autofunction:: InverseMatrixMultiplication

   Matrix inverse multiplication. Used by MBAT for unbinding.

.. autofunction:: ElementAngleAddition

   Element-wise modular angle addition. Used by FHRR for binding.

.. autofunction:: ElementAngleSubtraction

   Element-wise modular angle subtraction. Used by FHRR for unbinding.

pyhdc.components.bundling
--------------------------

.. currentmodule:: pyhdc.components.bundling

.. autofunction:: ElementAddition

   Element-wise sum; no normalisation.

.. autofunction:: ElementAdditionCut

   Element-wise sum then clip to [-1, 1]. Used by MAP_C.

.. autofunction:: ElementAdditionBits

   Element-wise sum in a wide accumulator, then a single saturating clip to the
   integer bit-width range. Used by MAP_I_Bits.

.. autofunction:: ElementAdditionBinaryThreshold

   Majority-vote threshold. Used by BSC.

.. autofunction:: ElementAdditionBipolarThreshold

   Sum then sign function. Result in {-1, +1}.

.. autofunction:: ElementAdditionNormalized

   Sum then L2 normalise. Used by HRR, VTB, MBAT.

.. autofunction:: ElementAdditionConstantNormalized

   Sum then divide by √M. Used by HRR_ConstNorm.

.. autofunction:: AnglesOfElementAddition

   Phasor resultant angle. Used by FHRR.

.. autofunction:: Disjunction

   Bitwise OR. Used by BSDC family (without thinning).

.. autofunction:: DisjunctionThinned

   Bitwise OR then random thinning to a maximum density. Used by BSDC_THIN.

   :param density: Maximum output density (fraction of 1-bits) after thinning (float, default 0.5).

pyhdc.components.elements
--------------------------

.. currentmodule:: pyhdc.components.elements

Element generators control how individual hypervector values are drawn.
Each function has signature ``(size, dtype) -> ndarray``.

.. autofunction:: UniformBipolar

   Uniform random from {-1, +1}.

.. autofunction:: UniformAngles

   Uniform random in [-π, π).

.. autofunction:: NormalReal

   Standard normal N(0, 1).

.. autofunction:: BernoulliBinary

   Bernoulli(p=0.5) → {0, 1}.

.. autofunction:: BernoulliBipolar

   Bernoulli(p=0.5) → {-1, +1}.

.. autofunction:: BernoulliSparse

   k-sparse binary: exactly k elements are 1, rest 0.

   :param k: Number of 1s. Default: determined by the encoding.

.. autofunction:: SparseSegmented

   Segment-wise sparse binary: k ones per segment.

   :param segment_size: Number of elements per segment.
   :param k: Number of 1s per segment.

pyhdc.components.thinning
--------------------------

.. currentmodule:: pyhdc.components.thinning

.. autofunction:: NoThin

   No-op thinning function. Returns the input unchanged.
   Used by all encodings that do not perform thinning.

pyhdc.components.unary
--------------------------

.. currentmodule:: pyhdc.components.unary

Added in 2.1.0. Unary components back the four single-operand operations
(:func:`permute`, :func:`inverse`, :func:`negative`, :func:`normalize`)
exposed on :class:`~pyhdc.Hypervector`, on each encoding, and at module level.
Each function takes raw array data, operates **dimension-first** (axis 0 is the
hypervector dimension ``D``, trailing axes are the batch) and runs on both
numpy and torch backends.

An encoding wires these into its ``EncodingSpec`` through the ``permute_fn``,
``inverse_fn``, ``normalize_fn``, and ``negative_fn`` fields. ``permute_fn``
defaults to :func:`CyclicShift` when left unset, so every encoding has a
permutation. The other three default to ``RaiseNotImplementedError``: an
encoding that does not assign ``inverse_fn`` / ``normalize_fn`` / ``negative_fn``
raises ``NotImplementedError`` when that operation is called. See
:doc:`encodings` for the per-family support of each operation.

.. autofunction:: CyclicShift

   Cyclic-shift permutation along axis 0, broadcasts over trailing batch axes.
   Encoding-agnostic, so it is the default ``permute_fn`` for all 15 encodings.
   Implemented as ``np.roll(data, shift, axis=0)`` (torch:
   ``torch.roll(data, shift, dims=0)``). A negative ``shift`` is the exact
   inverse of the positive shift.

   :param shift: Number of positions to shift. Default ``1``.

.. autofunction:: IdentityInverse

   Returns ``data`` unchanged. The binding inverse for self-inverse schemes
   where each element is its own inverse: MAP bipolar multiply and BSC XOR.

.. autofunction:: ReverseInverse

   Exact involution inverse of circular convolution, used by the HRR family.
   Keeps index 0 and reverses the remaining coordinates along axis 0:
   ``np.concatenate([data[:1], np.flip(data[1:], axis=0)], axis=0)`` (torch:
   ``torch.cat([data[:1], torch.flip(data[1:], dims=[0])], dim=0)``).

.. autofunction:: PhaseNegate

   FHRR binding inverse: negate the phase modulo 2π, ``np.mod(-data, 2*pi)``.

.. autofunction:: Negate

   Additive (bundling) inverse: element-wise negation ``-data``.

.. autofunction:: L2Normalize

   Normalise each hypervector to unit L2 length along axis 0,
   ``data / np.linalg.norm(data, axis=0, keepdims=True)``. The canonical form
   for HRR, VTB, and MBAT.

.. autofunction:: WrapPhase

   Normalise FHRR phases to the entry distribution range ``[-pi, pi)``,
   ``np.mod(data + pi, 2*pi) - pi``.

.. autofunction:: SignNormalize

   Normalise MAP hypervectors back to bipolar ``{-1, 0, +1}`` by sign,
   ``np.sign(data)``.

The table maps each unary operation to the component that backs it and the
encoding families that use it.

.. list-table::
   :header-rows: 1

   * - Operation
     - Component
     - Used by
   * - ``permute``
     - :func:`CyclicShift`
     - All 15 encodings (default fallback)
   * - ``inverse``
     - :func:`IdentityInverse`
     - MAP_I, MAP_I_Bits, MAP_B, BSC
   * - ``inverse``
     - :func:`ReverseInverse`
     - HRR, HRR_NoNorm, HRR_ConstNorm
   * - ``inverse``
     - :func:`PhaseNegate`
     - FHRR
   * - ``negative``
     - :func:`Negate`
     - MAP_C, MAP_I, MAP_I_Bits, MAP_B, HRR, HRR_NoNorm, HRR_ConstNorm, VTB, MBAT
   * - ``normalize``
     - :func:`L2Normalize`
     - HRR, HRR_NoNorm, HRR_ConstNorm, VTB, MBAT
   * - ``normalize``
     - :func:`WrapPhase`
     - FHRR
   * - ``normalize``
     - :func:`SignNormalize`
     - MAP_C, MAP_I, MAP_I_Bits, MAP_B
