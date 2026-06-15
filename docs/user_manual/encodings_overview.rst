Encodings Overview
==================

An *encoding* in PyHDC is a complete specification of how hypervectors are
generated and how the three primitives (bundle, bind, similarity) are
implemented. The sections below cover the shared base class and then each
of the four families.

The Encoding base class
------------------------

All encoding classes inherit from :class:`~pyhdc.Encoding`. The constructor
accepts these shared parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``dimension``
     - int
     - Number of elements per hypervector (default: 10,000)
   * - ``backend``
     - str
     - ``"numpy"`` (default) or ``"torch"``
   * - ``device``
     - str or None
     - PyTorch device string (e.g., ``"cuda"``, ``"cpu"``); ignored for NumPy
   * - ``dtype``
     - dtype or None
     - Override the default data type
   * - ``mask``
     - int or None
     - Bit mask for MAP_I_Bits; sets the integer bit width
   * - ``generator``
     - HDCGenerator or None
     - Custom random generator; if None, uses NumPy's default RNG
   * - ``similarity_remap``
     - callable or None
     - Function applied to every similarity result (e.g., ``remap_to_unit``)

The encoding delegates to an ``EncodingSpec`` dataclass that wires together
the five component functions:

.. code-block:: python

   @dataclass
   class EncodingSpec:
       dtype: Any
       element_generator: Callable
       similarity_fn: Callable
       bundling_fn: Callable
       thinning_fn: Callable
       binding_fn: Callable
       unbinding_fn: Callable
       mask: Optional[int] = None
       generator_output_type: Literal["bits", "words", "floats"] = "floats"

Subclasses implement ``_get_encoding_spec()`` to return a populated
``EncodingSpec``. Users never interact with ``EncodingSpec`` directly.

MAP family
-----------

Multiplicative-Additive-Permutation encodings use **dense bipolar** or
**binary** vectors.

**MAP_C**; Continuous MAP
   * Elements: float in [-1, 1], default dtype ``float32``
   * Binding: element-wise multiplication (self-inverse)
   * Bundling: element-wise addition with threshold/cut and band randomisation
   * Similarity: cosine
   * Unbind: yes
   * Best for: general-purpose, well-studied, good continuous capacity

**MAP_I**; Integer MAP
   * Elements: int {-1, +1}, default dtype ``int32``
   * Same operations as MAP_C but in integer arithmetic
   * Requires a bit-output or word-output generator
   * Unbind: yes

**MAP_I_Bits**; Fixed-Width Integer MAP
   * Like MAP_I but the ``mask`` parameter sets a custom integer bit width
   * Useful for hardware implementations targeting specific word sizes

**MAP_B**; Binary MAP
   * Elements: binary {0, 1}, default dtype ``int8``
   * Dense binary variant; binding is element-wise product; bundling clips to {0, 1}
   * Unbind: yes

HRR family
-----------

Holographic Reduced Representations use **dense continuous** vectors with
circular convolution binding.

**HRR**; Holographic Reduced Representation
   * Elements: normal float, default dtype ``float32``
   * Binding: circular convolution (implemented via FFT)
   * Unbinding: circular correlation
   * Bundling: element-wise addition then L2 normalisation
   * Similarity: cosine
   * Best for: theoretical cleanliness; the canonical VSA for research papers

**HRR_NoNorm**
   * Like HRR but bundling does not normalise the result
   * Vector magnitude grows with the number of bundled items

**HRR_ConstNorm**
   * Bundling normalises by :math:`\sqrt{M}` where :math:`M` is the number
     of bundled vectors, keeping constant norm regardless of bundle size

**FHRR**; Fourier HRR
   * Elements: angles in [0, 2π], stored as float32
   * Binding: element-wise angle addition (modular arithmetic)
   * Unbinding: element-wise angle subtraction
   * Bundling: compute resultant angle of summed phasors
   * Similarity: cosine of element-wise angle difference
   * Best for: periodic signals, phase-based feature spaces

Matrix family
--------------

These encodings use matrix operations for binding, giving them stronger
algebraic properties at the cost of additional storage.

**VTB**; Vector-derived Transformation Binding
   * Elements: normal float, dtype ``float32``
   * Binding: constructs a matrix from the key vector and applies it to the value
   * Unbinding: transpose of the transformation matrix
   * Bundling: normalised addition (same as HRR)
   * Best for: replicating VTB literature results; theoretically motivated
     matrix binding

**MBAT**; Matrix Binding of Additive Terms
   * Elements: normal float, dtype ``float32``
   * Binding: multiply by a random matrix; the matrix is stored in metadata
   * Unbinding: multiply by the matrix inverse (retrieved from ``get_metadata()``)
   * Important: you must preserve the metadata dict from the binding result
     to perform unbinding later

Binary family
--------------

**BSC**; Binary Spatter Code
   * Elements: binary {0, 1}, default dtype ``int8``
   * Dense (Bernoulli p = 0.5)
   * Binding: XOR (exactly self-inverse)
   * Bundling: majority-vote threshold (element is 1 if > half the inputs are 1)
   * Similarity: Hamming distance (remapped to [-1, 1])
   * Unbind: yes (exact, not approximate)
   * Best for: hardware efficiency; situations where exact unbinding is needed

Sparse Binary (BSDC) family
-----------------------------

All BSDC variants share:

* Elements: sparse binary {0, 1}, dtype ``int8``
* Initial density ≈ 1–5% (controlled by ``BernoulliSparse`` element generator)
* Bundling: bitwise OR (with or without thinning)
* Similarity: Overlap (remapped to [-1, 1])

**BSDC_CDT**; Context-Dependent Thinning
   * Binding: Additive context-dependent thinning
   * Unbind: **not supported** (thinning is not invertible)
   * Bundling: OR only (no thinning); density grows with each bundle step
   * Best for: papers that specifically use CDT binding

**BSDC_S**; Sparse with Shifting
   * Binding: circular shift by one position per bind step
   * Unbind: yes (inverse shift)
   * Bundling: OR only; density grows
   * Best for: positional / sequential encodings

**BSDC_SEG**; Sparse Segmented
   * Like BSDC_S but shift is applied per-segment of the vector
   * Useful for segment-wise positional encoding

**BSDC_THIN**; Sparse with Thinning (v1.1.0)
   * Binding: circular shift (same as BSDC_S)
   * Unbind: yes
   * Bundling: OR followed by random thinning to maintain target density
   * Best for: applications requiring many bundle steps without density saturation;
     the recommended sparse default
