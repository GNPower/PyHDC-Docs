The components Submodule
========================

``pyhdc.components`` exposes the individual building blocks that underpin
every encoding. Most users never need this module: the encoding classes
assemble the right components automatically. It is useful when:

* You are writing a custom encoding subclass
* You want to apply a specific operation (e.g., ``remap_to_unit``) without
  a full encoding context
* You are debugging or testing individual operations in isolation

Submodule layout
-----------------

.. code-block:: text

   pyhdc.components
   ├── binding          : all binding functions
   ├── bundling         : all bundling functions
   ├── similarity       : all similarity functions + remap_to_unit
   ├── elements         : element generator functions (how random values are drawn)
   ├── thinning         : thinning functions (post-process sparse binary vectors)
   ├── unary            : permutation and per-vector unary functions (inverse, negative, normalize)
   ├── basis            : family-aware basis builders (codebooks for data encoders)
   ├── quantization     : sign and tanh quantization of raw element values
   └── input_formatting : internal normalisation utilities

The EncodingSpec wiring
------------------------

When you define a custom encoding by subclassing :class:`~pyhdc.Encoding` and
implementing ``_get_encoding_spec()``, you return an ``EncodingSpec`` that
names the component functions to use:

.. code-block:: python

   from pyhdc.encodings.base import Encoding, EncodingSpec
   from pyhdc.components.binding   import ElementMultiplication
   from pyhdc.components.bundling  import ElementAdditionCut
   from pyhdc.components.similarity import CosineSimilarity
   from pyhdc.components.elements  import UniformBipolar
   from pyhdc.components.thinning  import NoThin
   import numpy as np

   class MyEncoding(Encoding):
       def _get_encoding_spec(self) -> EncodingSpec:
           return EncodingSpec(
               dtype=np.float32,
               element_generator=UniformBipolar,
               similarity_fn=CosineSimilarity,
               bundling_fn=ElementAdditionCut,
               thinning_fn=NoThin,
               binding_fn=ElementMultiplication,
               unbinding_fn=ElementMultiplication,   # self-inverse
               generator_output_type="floats",
           )

elements submodule
-------------------

Element generators control how individual hypervector values are drawn.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``UniformBipolar``
     - Uniform random from {-1, +1} (Bernoulli p=0.5 then x2-1)
   * - ``UniformAngles``
     - Uniform random in [-π, π)
   * - ``NormalReal``
     - Normal distribution N(0, 1)
   * - ``BernoulliBinary``
     - Bernoulli(p=0.5) -> {0, 1}
   * - ``BernoulliBipolar``
     - Bernoulli(p=0.5) -> {-1, +1}
   * - ``BernoulliSparse``
     - k-sparse binary: exactly k elements are 1, rest are 0
   * - ``SparseSegmented``
     - Per-segment sparse binary: k ones placed uniformly within each segment

thinning submodule
-------------------

Thinning operations post-process a bundled binary hypervector to reduce
density.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``NoThin``
     - No-op; returns the input unchanged. Used by encodings that do not thin.

unary submodule
---------------

Added in 2.1.0. The ``pyhdc.components.unary`` module holds the permutation
function and the per-vector unary functions. Each takes raw array data,
operates dimension-first (axis 0 is the hypervector dimension ``D``), and
works on both numpy and torch backends. An ``EncodingSpec`` wires these into
the ``permute_fn``, ``inverse_fn``, ``negative_fn``, and ``normalize_fn``
fields, leaving ``permute_fn`` at ``None`` selects the shared ``CyclicShift``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``CyclicShift``
     - Cyclic-shift permutation along axis 0. Broadcasts over trailing batch axes. The default ``permute`` for every encoding.
   * - ``IdentityInverse``
     - Returns the input unchanged, the binding inverse for self-inverse schemes (MAP bipolar multiply, BSC XOR).
   * - ``ReverseInverse``
     - Exact involution inverse of circular convolution (HRR). Keeps index 0 and reverses the remaining coordinates along axis 0.
   * - ``PhaseNegate``
     - FHRR binding inverse, negates the phase modulo 2π.
   * - ``Negate``
     - Additive (bundling) inverse, element-wise negation ``-data``.
   * - ``L2Normalize``
     - Normalizes each hypervector to unit L2 length along axis 0.
   * - ``WrapPhase``
     - Normalizes FHRR phases to the canonical range [-π, π).
   * - ``SignNormalize``
     - Normalizes MAP hypervectors back to bipolar {-1, 0, +1} by sign.

basis submodule
---------------

Added in 2.2.0. The ``pyhdc.components.basis`` package holds the family-aware
builders that produce a codebook for a given encoding. Each builder has the
signature ``fn(encoding, count, dimension=None)`` and returns a ``(D, count)``
array in the encoding's value domain and backend. These builders back the
codebook data encoders (``Level``, ``Thermometer``, ``Circular``, etc.),
see :doc:`data_encoders`.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``empty``
     - ``count`` all-zero hypervectors as a ``(D, count)`` array.
   * - ``identity``
     - ``count`` copies of the binding-identity element ``e`` (where ``bind(x, e) == x``).
   * - ``random``
     - ``count`` independent random hypervectors as a ``(D, count)`` codebook.
   * - ``level``
     - A linear level codebook: adjacent columns correlated, ends near-orthogonal.
   * - ``circular``
     - A ring level codebook: similarity wraps, so level 0 ~ level L-1.
   * - ``thermometer``
     - A deterministic cumulative (unary) codebook. Discrete families only.

Two scalar helpers sit alongside the builders:

* ``family_endpoints(encoding)`` : the ``(low, high)`` element endpoints for the
  encoding's value domain.
* ``binding_identity(encoding, dimension=None)`` : the binding-identity element
  ``e`` as a ``(D,)`` array.

quantization submodule
----------------------

Added in 2.2.0. The ``pyhdc.components.quantization`` module maps continuous
element values to a bipolar form. Both functions take a raw array, operate
dimension-first (axis 0 is the hypervector dimension ``D``), and work on the
numpy and torch backends.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``hard_quantize(data)``
     - Maps each value to ``{-1, 0, +1}`` by sign.
   * - ``soft_quantize(data, temperature=1.0)``
     - Smooth bipolar surrogate ``tanh(data / temperature)`` in ``(-1, 1)``.

bundling helpers
----------------

Added in 2.2.0. The ``pyhdc.components.bundling`` module now has composable helpers
that reduce a stacked set without a family-specific threshold or thinning step.
For family-aware bundling, use ``Encoding.bundle``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``randsel(data, p=None)``
     - Random-selection bundling: copy each coordinate from one randomly chosen input column. ``p`` is an optional length-``N`` weight vector (default uniform, normalized internally).
   * - ``multirandsel(data, count, p=None)``
     - ``count`` independent ``randsel`` draws as a ``(D, count)`` array.
   * - ``multiset(data, axis=-1)``
     - Additive multiset: sum a stacked ``(D, N)`` set into a single ``(D,)`` vector over ``axis``.
   * - ``multibundle(data, axis=-1)``
     - Alias of ``multiset`` under its conventional HDC name.

binding helper
--------------

Added in 2.2.0. The ``pyhdc.components.binding`` module gains a product-reduce
helper. For non-multiplicative binders, use ``Encoding.bind``.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``multibind(data, axis=-1)``
     - Multiplicative bind of a stacked ``(D, N)`` set into a single ``(D,)`` vector (product over ``axis``).

similarity submodule
---------------------

The similarity module exports the four metric functions and the remap utility:

* ``CosineSimilarity(*hvs)`` : cosine similarity
* ``HammingDistance(*hvs)`` : normalised Hamming, output in [-1, 1]
* ``Overlap(*hvs)`` : normalised overlap, output in [-1, 1]
* ``AngleDistance(*hvs)`` : angle-based distance, output in [-1, 1]
* ``remap_to_unit(sim)`` : maps [-1, 1] -> [0, 1]

Each function accepts one or two arguments in the same calling conventions
as the ``Encoding.similarity()`` method. Each metric also takes a ``mode`` of
``"pairwise"`` or ``"cross"`` and a keyword-only ``axis``, see
:doc:`similarity_metrics`.

input_formatting submodule
---------------------------

Internal utilities used by encoding methods to normalise inputs. These are
considered private API and may change between releases:

* ``_extract_data(hv)`` : extract the raw array from a Hypervector or pass
  through if already an array
* ``_normalize_inputs(*hvs)`` : validate and normalise a sequence of inputs
* ``_detect_batch_structure(*hvs)`` : determine whether inputs are single
  ``(D,)`` vectors or ``(D, N)`` batches and which column-wise calling
  convention applies
