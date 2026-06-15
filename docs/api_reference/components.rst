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
             Supports three :ref:`batched calling conventions <similarity-batched>`.

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

   Element-wise sum with integer bit-width clipping. Used by MAP_I_Bits, MAP_B.

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

   Bitwise OR then random thinning to maintain target density. Used by BSDC_THIN.

   :param target_density: Target density after thinning (float, 0 < d < 1).

pyhdc.components.elements
--------------------------

.. currentmodule:: pyhdc.components.elements

Element generators control how individual hypervector values are drawn.
Each function has signature ``(size, dtype) -> ndarray``.

.. autofunction:: UniformBipolar

   Uniform random from {-1, +1}.

.. autofunction:: UniformAngles

   Uniform random in [0, 2π].

.. autofunction:: NormalReal

   Standard normal N(0, 1).

.. autofunction:: BernoulliBinary

   Bernoulli(p=0.5) → {0, 1}.

.. autofunction:: BernoulliBiploar

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
