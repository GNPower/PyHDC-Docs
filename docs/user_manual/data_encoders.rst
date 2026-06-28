Data Encoders
=============

An *encoding* fixes the algebra (bundle, bind, similarity), while a *data encoder*
turns raw data into a hypervector. An :class:`~pyhdc.Encoder` wraps one 
:class:`~pyhdc.Encoding` instance and reuses its ``generate``, ``normalize_fn``, 
backend, and device. The output is a :class:`~pyhdc.Hypervector` in the given 
encoding, so it flows straight into :func:`~pyhdc.bundle`, :func:`~pyhdc.bind`, 
:func:`~pyhdc.similarity`, and :meth:`~pyhdc.Hypervector.select` with no conversion step.

Object model
------------

Every encoder is dimension-first, the same convention as the rest of PyHDC: a
scalar encodes to a ``(D,)`` hypervector, a batch of ``B`` values to ``(D, B)``.
``encode(value)`` and calling the encoder directly are equivalent, so
``enc(0.5)`` and ``enc.encode(0.5)`` are the same. The ``params`` property exposes the encoder's
parameter array: the precomputed basis for a codebook encoder, or the projection
or weight array for a functional one.

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_I(dimension=10_000)
   level = pyhdc.Level(enc, levels=100, low=0.0, high=1.0)

   a     = level.encode(0.30)              # (D,) Hypervector
   b     = level.encode(0.32)              # a close value
   c     = level.encode(0.90)              # a far value
   batch = level.encode([0.1, 0.5, 0.9])   # (D, 3) Hypervector

   a.data.shape                # (10000,)
   batch.data.shape            # (10000, 3)
   a.similarity(b)             # ~0.98  (near)
   a.similarity(c)             # ~0.41  (far)
   level.params.shape          # (10000, 100)

The two families differ in what ``params`` holds and how ``encode`` reads it.

Codebook family
---------------

The codebook encoders are :class:`~pyhdc.Empty`, :class:`~pyhdc.Identity`,
:class:`~pyhdc.Random`, :class:`~pyhdc.Level`, :class:`~pyhdc.Thermometer`, and
:class:`~pyhdc.Circular`. Each holds a precomputed ``(D, L)`` basis built by a
:mod:`pyhdc.components.basis` builder. ``encode`` maps a value to the nearest level
index (clamp the value into range, then quantize to the closest of ``L`` levels)
and selects that column. A batch of values selects a batch of columns.

The constructor signature is ``(encoding, levels, low=0.0, high=1.0)``. ``levels``
must be at least 1 and ``high`` must be greater than ``low``, or the constructor
raises ``ValueError``.

:class:`~pyhdc.Circular` is the exception to the clamp rule: it wraps the index
modulo ``levels`` instead of clamping, so the top of the range rejoins the bottom.
Use it for periodic values such as hour of day or compass heading.

Functional family
------------------

The functional encoders are :class:`~pyhdc.Projection`,
:class:`~pyhdc.Sinusoid`, :class:`~pyhdc.Density`, and
:class:`~pyhdc.FractionalPower`. Instead of indexing a codebook, each maps a
feature vector ``(F,)`` (or a batch ``(F, B)``) to ``(D, B)``. ``Projection`` and
``Sinusoid`` take ``(encoding, features)``, ``Density`` takes
``(encoding, low=0.0, high=1.0)``, ``FractionalPower`` takes ``(encoding)`` and
raises a base atom to a per-value fractional power.

Per-family support
------------------

Each encoder defines its mapping only where the family's algebra supports it. An
unsupported pairing raises ``NotImplementedError`` at construction rather than
returning a wrong result.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Encoder
     - Family support
   * - ``Empty``, ``Random``, ``Level``, ``Circular``
     - Family-agnostic, defined for every encoding.
   * - ``Identity``
     - Raises for VTB, MBAT, and the BSDC family (no neutral binding element).
   * - ``Thermometer``, ``Density``
     - Discrete families only (MAP_I, MAP_B, BSC, BSDC), raise on continuous and
       phase families.
   * - ``Projection``
     - Needs a family with a normalize step (MAP, HRR family, VTB, MBAT, FHRR),
       raises on BSC and the BSDC family.
   * - ``Sinusoid``
     - No family gate. It builds on any encoding, but its real-valued output suits
       the cosine and HRR families rather than FHRR phase vectors.
   * - ``FractionalPower``
     - Defined only for FHRR (phase scaling) and the HRR family (FFT), raises
       elsewhere.

Family-aware basis builders
---------------------------

The codebook a codebook encoder holds comes from :mod:`pyhdc.components.basis`.
That package exposes :func:`~pyhdc.components.basis.empty`,
:func:`~pyhdc.components.basis.identity`, :func:`~pyhdc.components.basis.random`,
:func:`~pyhdc.components.basis.level`, :func:`~pyhdc.components.basis.circular`,
and :func:`~pyhdc.components.basis.thermometer`, each with the signature
``fn(encoding, count, dimension=None)`` returning a ``(D, count)`` array in the
encoding's value domain and backend. These are the same builders the codebook
encoders hold as ``params``: ``Level(enc, levels=100).params`` is the array that
``pyhdc.components.basis.level(enc, 100)`` returns.

Two domain helpers sit alongside the builders.
:func:`~pyhdc.components.basis.family_endpoints` returns the ``(low, high)``
element endpoints of a discrete family's value domain (``thermometer`` and
``Density`` use it). :func:`~pyhdc.components.basis.binding_identity` returns the
binding-identity element ``e`` (where ``bind(x, e) == x``) as a ``(D,)`` array.

``Projection`` and ``FractionalPower`` reuse the encoding's normalize step, the
component behind it is described on :doc:`unary_operations`. For the building
blocks these encoders draw on, see :doc:`components_overview`, for the algebra each
encoding fixes, see :doc:`encodings_overview`.

Top-level re-exports
--------------------

The encoder classes live in :mod:`pyhdc.encoders` and are re-exported at the top
level, so ``pyhdc.Level``, ``pyhdc.Projection``, and the rest are available
directly. ``pyhdc.Level`` is ``pyhdc.encoders.Level``.
