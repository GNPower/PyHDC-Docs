Data Encoders
=============

.. currentmodule:: pyhdc

Data encoders map raw data (a scalar, a feature vector, or a batch) into a
:class:`Hypervector`. Each wraps an :class:`Encoding` and reuses its
``generate`` / ``normalize_fn`` / backend, so encoder output works with the existing
bundle / bind / similarity / select functions. Everything is dimension-first: a batch
of ``B`` inputs encodes to a ``(D, B)`` hypervector.

Encoder base class
------------------

.. autoclass:: Encoder
   :members:
   :show-inheritance:

Codebook encoders
-----------------

A value indexes into a precomputed ``(D, L)`` basis. Constructor signature
``(encoding, levels, low=0.0, high=1.0)``: ``levels`` must be at least 1 and ``high``
must exceed ``low``, or the constructor raises ``ValueError``.

.. autoclass:: Empty
   :show-inheritance:

   Every value maps to the all-zero hypervector. A structural placeholder, useful
   as a neutral element or for testing.

.. autoclass:: Identity
   :show-inheritance:

   Every value maps to the binding-identity element ``e`` (the vector where
   ``bind(x, e) == x``): all-ones for MAP, all-zeros for BSC, the impulse for the HRR
   family, zero phase for FHRR. Raises ``NotImplementedError`` for VTB, MBAT, and the
   BSDC family, which have no neutral binding element.

.. autoclass:: Random
   :show-inheritance:

   Each level is an independent random hypervector, an unstructured codebook of
   ``levels`` near-orthogonal atoms.

.. autoclass:: Level
   :show-inheritance:

   A linear level code: nearby values map to correlated hypervectors and distant
   values to near-orthogonal ones, so similarity falls off monotonically with value
   distance.

.. autoclass:: Thermometer
   :show-inheritance:

   A cumulative (thermometer) code: set coordinates accumulate as the value rises, so
   every level includes the ones below it. Discrete families only, raises
   ``NotImplementedError`` for continuous and phase families.

.. autoclass:: Circular
   :show-inheritance:

   A ring level code for periodic values: the index wraps modulo ``levels``, so the
   top of the range rejoins the bottom and similarity is periodic.

Functional encoders
-------------------

A transform of a feature vector.

.. autoclass:: Projection
   :show-inheritance:

   Random linear projection ``W @ x`` followed by the encoding's ``normalize_fn``.
   Needs a family that defines a normalize step (MAP, HRR family, VTB, MBAT, FHRR);
   raises ``NotImplementedError`` for BSC and the BSDC family.

.. autoclass:: Sinusoid
   :show-inheritance:

   Random Fourier features ``cos(W @ x + b) * sqrt(2 / D)``, a real-valued kernel
   approximation. Pairs with the cosine and HRR families rather than FHRR phase
   vectors.

.. autoclass:: Density
   :show-inheritance:

   A population (threshold) code over a scalar in ``[low, high]``. Discrete families
   only, raises ``NotImplementedError`` for continuous and phase families.

.. autoclass:: FractionalPower
   :show-inheritance:

   The fractional binding power of a base vector. Defined only for the FHRR (phase
   scaling) and the HRR families (FFT), raises ``NotImplementedError`` elsewhere.
