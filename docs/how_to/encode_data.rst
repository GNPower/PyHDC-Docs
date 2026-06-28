How to Encode Data into Hypervectors
====================================

An *encoding* fixes the algebra (bundle, bind, similarity). A *data encoder*
turns raw values into hypervectors. PyHDC provides two families:

* **Codebook encoders** (``Level``, ``Circular``, ``Thermometer``, ``Empty``,
  ``Identity``, ``Random``) hold a precomputed ``(D, L)`` basis. A value picks the
  nearest level and the encoder returns that column.
* **Functional encoders** (``Projection``, ``Sinusoid``, ``Density``,
  ``FractionalPower``) transform a feature vector.

Every encoder wraps one :class:`~pyhdc.Encoding` instance and is dimension-first:
a scalar encodes to ``(D,)``, a batch of ``B`` values to ``(D, B)``.
``encoder.encode(value)`` and ``encoder(value)`` are the same call. For the
concepts behind the two families, see :doc:`../user_manual/data_encoders`.

Encode a scalar with Level
--------------------------

:class:`~pyhdc.Level` spaces ``levels`` hypervectors so that nearby values map to
correlated codes and distant values to near-orthogonal ones:

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_I(dimension=10_000)
   level = pyhdc.Level(enc, levels=100, low=0.0, high=1.0)

   hv = level.encode(0.5)   # one (10000,) hypervector
   print(hv.shape)          # (10000,)

Similarity to a fixed hypervector falls monotonically as the value moves away:

.. code-block:: python

   zero = level.encode(0.0)
   print(enc.similarity(zero, level.encode(0.1)))   # ~= 0.90
   print(enc.similarity(zero, level.encode(0.5)))   # ~= 0.51
   print(enc.similarity(zero, level.encode(1.0)))   # ~= 0.02

Values outside ``[low, high]`` clamp to the nearest endpoint.

Encode many values at once
--------------------------

Pass a list (or any 1-D array) to encode a whole batch in one call. Each value
becomes one column:

.. code-block:: python

   batch = level.encode([0.0, 0.25, 0.5, 0.75, 1.0])
   print(batch.shape)   # (10000, 5)

Periodic values with Circular
-----------------------------

:class:`~pyhdc.Circular` wraps the level index modulo ``levels``, so the top of
the range rejoins the bottom. Use it for angles, hours, days, or any cyclic
quantity:

.. code-block:: python

   hours = pyhdc.Circular(enc, levels=24, low=0.0, high=24.0)

   print(enc.similarity(hours.encode(23.0), hours.encode(0.0)))   # ~= 0.9  (adjacent across the wrap)
   print(enc.similarity(hours.encode(0.0),  hours.encode(12.0)))  # ~= 0.0  (opposite on the ring)

Discrete encoders: Thermometer and Density
-------------------------------------------

:class:`~pyhdc.Thermometer` (a cumulative code) and :class:`~pyhdc.Density` (a
population code) are defined for the discrete families (MAP_I, MAP_B, BSC, and the
BSDC family). Build them on a discrete encoding:

.. code-block:: python

   binary = pyhdc.MAP_B(dimension=10_000)

   therm = pyhdc.Thermometer(binary, levels=20, low=0.0, high=1.0)
   dens  = pyhdc.Density(binary, low=0.0, high=1.0)
   print(therm.encode(0.5).shape)   # (10000,)
   print(dens.encode(0.5).shape)    # (10000,)

Constructing either on a continuous or phase family (MAP_C, the HRR family, VTB,
MBAT, FHRR) raises ``NotImplementedError``, because those domains have no two
endpoint elements to interpolate between.

Encode a feature vector with Projection
---------------------------------------

:class:`~pyhdc.Projection` applies a fixed random linear map to a length-``F``
feature vector, then the encoding's normalize step. It accepts a single vector or
a batch:

.. code-block:: python

   import numpy as np

   proj = pyhdc.Projection(enc, features=8)
   print(proj.encode(np.random.rand(8)).shape)      # (10000,)
   print(proj.encode(np.random.rand(8, 5)).shape)   # (10000, 5)

:class:`~pyhdc.Sinusoid` is the related random-Fourier-feature map for the cosine
and HRR families. ``Projection`` needs a family with a normalize step
(MAP, HRR, VTB, MBAT, FHRR) and raises ``NotImplementedError`` on BSC and BSDC.

``Empty``, ``Random``, and ``Identity`` round out the set as structural codebooks:
all-zero, independent-random, and the binding-identity element respectively. They
take the same ``(encoding, levels, low, high)`` constructor as ``Level``.

Putting it together: a value-feature classifier
------------------------------------------------

Encoders compose with the core operations. The pattern below binds each feature
value to a per-feature key, bundles the bound pairs into one record hypervector,
bundles records into a class prototype, and classifies a whole test batch in a
single :doc:`cross-similarity <compute_similarity>` call:

.. code-block:: python

   import numpy as np
   import pyhdc

   np.random.seed(0)
   enc = pyhdc.MAP_I(dimension=10_000)

   F     = 4                                    # features per record
   keys  = [enc.generate() for _ in range(F)]   # one key hypervector per feature
   value = pyhdc.Level(enc, levels=64, low=0.0, high=1.0)

   def encode_record(row):
       pairs = [pyhdc.bind(keys[i], value.encode(float(row[i]))) for i in range(F)]
       return pyhdc.bundle(*pairs)

   rng    = np.random.default_rng(0)
   class0 = rng.normal(0.3, 0.05, size=(3, F))   # a low-valued class
   class1 = rng.normal(0.7, 0.05, size=(3, F))   # a high-valued class
   protos = pyhdc.stack([
       pyhdc.bundle(*[encode_record(r) for r in class0]),
       pyhdc.bundle(*[encode_record(r) for r in class1]),
   ])                                            # (10000, 2)

   test   = np.vstack([rng.normal(0.3, 0.05, size=(3, F)),
                       rng.normal(0.7, 0.05, size=(3, F))])
   testhv = pyhdc.stack([encode_record(r) for r in test])   # (10000, 6)

   scores = protos.similarity(testhv, mode="cross")         # (2, 6)
   print(np.asarray(scores).argmax(axis=0))                 # [0 0 0 1 1 1]

The first three test records score highest against prototype 0 and the last three
against prototype 1, which is the correct split.

See also
--------

* :doc:`choose_encoding` : which encoding family to build the encoder on, and the
  per-encoder family constraints.
* :doc:`compute_similarity` : pairwise and cross similarity, including the
  ``mode="cross"`` matrix used above.
* :doc:`../user_manual/data_encoders` : the encoder object model and the
  family-aware basis builders.
