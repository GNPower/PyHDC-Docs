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
     - Uniform random in [0, 2π]
   * - ``NormalReal``
     - Normal distribution N(0, 1)
   * - ``BernoulliBinary``
     - Bernoulli(p=0.5) -> {0, 1}
   * - ``BernoulliBiploar``
     - Bernoulli(p=0.5) -> {-1, +1}  *(note: typo in source; "Biploar")*
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

similarity submodule
---------------------

The similarity module exports the four metric functions and the remap utility:

* ``CosineSimilarity(*hvs)`` : cosine similarity
* ``HammingDistance(*hvs)`` : normalised Hamming, output in [-1, 1]
* ``Overlap(*hvs)`` : normalised overlap, output in [-1, 1]
* ``AngleDistance(*hvs)`` : angle-based distance, output in [-1, 1]
* ``remap_to_unit(sim)`` : maps [-1, 1] -> [0, 1]

Each function accepts one or two arguments in the same calling conventions
as the ``Encoding.similarity()`` method.

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
