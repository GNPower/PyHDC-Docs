Changelog
=========

All notable changes to PyHDC are documented here. The project follows
`Semantic Versioning <https://semver.org/>`_ and
`Keep a Changelog <https://keepachangelog.com/>`_ conventions.

The source is `CHANGELOG.md on GitHub
<https://github.com/GNPower/PyHDC/blob/main/CHANGELOG.md>`_.

----

v2.2.0: 2026-06-27
---------------------

Added
~~~~~

* Data encoders in the new ``pyhdc.encoders`` package. Each :class:`~pyhdc.encoders.Encoder`
  wraps an :class:`~pyhdc.Encoding` instance and maps a value, feature vector, or batch to
  a dimension-first ``(D, B)`` :class:`~pyhdc.Hypervector` via ``encode`` (or by calling the
  encoder directly). Codebook encoders: :class:`~pyhdc.encoders.Empty`,
  :class:`~pyhdc.encoders.Identity`, :class:`~pyhdc.encoders.Random`,
  :class:`~pyhdc.encoders.Level`, :class:`~pyhdc.encoders.Thermometer`,
  :class:`~pyhdc.encoders.Circular`. Functional encoders:
  :class:`~pyhdc.encoders.Projection`, :class:`~pyhdc.encoders.Sinusoid`,
  :class:`~pyhdc.encoders.Density`, :class:`~pyhdc.encoders.FractionalPower`. A
  family-specific encoder raises ``NotImplementedError`` where the family has no definition
  (``Identity`` on VTB/MBAT/BSDC, ``Thermometer``/``Density`` on continuous or phase
  families, ``Projection`` on BSC/BSDC, ``FractionalPower`` outside FHRR and the HRR family).
  ``Identity`` returns the binding-identity element (the ``e`` where ``bind(x, e) == x``):
  all-ones for MAP, all-zeros for BSC, the impulse for the HRR family, zero phase for FHRR.
* Family-aware basis builders in the new ``pyhdc.components.basis`` package:
  :func:`~pyhdc.components.basis.empty`, :func:`~pyhdc.components.basis.identity`,
  :func:`~pyhdc.components.basis.random`, :func:`~pyhdc.components.basis.level`,
  :func:`~pyhdc.components.basis.circular`, :func:`~pyhdc.components.basis.thermometer`,
  plus :func:`~pyhdc.components.basis.family_endpoints`. Each returns a ``(D, count)``
  codebook in the encoding's value domain and backend.
* Cross similarity. :meth:`~pyhdc.Encoding.similarity` with ``mode="cross"``, ``A=(D, P)``
  and ``B=(D, M)`` returns the full ``(P, M)`` matrix of every column of ``A`` against every
  column of ``B``, backed by a single matmul with no ``(D, P, M)`` intermediate. Available on
  :meth:`~pyhdc.Encoding.similarity`, :meth:`~pyhdc.Hypervector.similarity`, and the new
  module-level :func:`~pyhdc.similarity`. Implemented for Cosine, Hamming, Overlap, and
  Angle; an encoding whose metric is outside that set raises ``NotImplementedError`` so the
  caller can fall back to a per-pair loop. Binary metrics cast to ``float64`` for a BLAS
  matmul, and cosine guards a zero-norm column (scores 0, not ``nan``).
* Module-level convenience function :func:`~pyhdc.similarity`, joining the existing
  :func:`~pyhdc.generate`, :func:`~pyhdc.zeros`, :func:`~pyhdc.bundle`, :func:`~pyhdc.bind`,
  and :func:`~pyhdc.unbind`.
* Composable component helpers, each in an operation-named module: random-selection bundling
  ``randsel`` / ``multirandsel`` and additive ``multiset`` / ``multibundle`` in
  ``pyhdc.components.bundling``, multiplicative ``multibind`` in
  ``pyhdc.components.binding``, and ``hard_quantize`` / ``soft_quantize`` in
  ``pyhdc.components.quantization``.
* ``MAP_I_Bits`` gains a ``bit_width`` parameter to set the signed saturation width
  explicitly (overrides ``mask``).

Changed (breaking)
~~~~~~~~~~~~~~~~~~~

* **Narrow:** ``MAP_I_Bits`` rejects a ``mask`` that is not of the form ``2**k - 1``
  (contiguous low bits). Such a value was previously accepted and silently ignored (always
  clipping at int32), it now raises ``ValueError``. Pass ``bit_width=k`` for an explicit
  k-bit limit. Default construction is unaffected.

Fixed
~~~~~

* :meth:`~pyhdc.Encoding.zeros` now works on the torch backend. It previously passed the
  encoding's numpy dtype straight to ``torch.zeros``, which raised a ``TypeError``, it now
  builds in numpy and converts, preserving the dtype.
* ``MAP_I_Bits`` now honors its bit width. The post-bundle saturation bounds and the storage
  dtype are derived from ``mask`` (which must be ``2**k - 1``) or the new ``bit_width``,
  instead of being hard-coded to int32 with the ``mask`` ignored. The default
  ``mask=(2**32) - 1`` is unchanged (int32 bounds, int32 storage). A narrow width now
  saturates correctly (an 8-bit mask clips to ``[-128, 127]`` and stores int8), a width
  wider than 32 widens the storage dtype (up to int64) so the sum no longer wraps on cast.

----

v2.1.0: 2026-06-18
---------------------

Added
~~~~~

* Multi-dimensional ``(D, N, M)`` batches. ``enc.generate(size=(D, N, M))`` returns
  one :class:`~pyhdc.Hypervector` wrapping a ``(D, N, M)`` array; axis 0 is the
  dimension ``D`` and every trailing-axis slice is a hypervector.
* ``axis=`` keyword on :meth:`~pyhdc.Encoding.bundle`: reduce a chosen batch axis (an
  int or a tuple of ints) and return a single :class:`~pyhdc.Hypervector`. ``axis=None``
  reduces the last axis, so ``(D, N)`` collapses to ``(D,)`` and ``(D, N, M)`` collapses
  to ``(D, N)``. Axis 0 is the dimension and cannot be reduced; passing ``axis=0`` raises
  ``ValueError``.
* ``axis=`` keyword (keyword-only) on :meth:`~pyhdc.Encoding.similarity`: for a single
  ``(D, N, M, ...)`` batch, selects which batch axis splits index 0 from the rest.
* :meth:`~pyhdc.Encoding.bind` and :meth:`~pyhdc.Encoding.unbind` batch automatically. The
  element-wise binders (MAP multiply, BSC xor, FHRR angle add/sub) broadcast a batch
  natively: a ``(D,)`` key binds against every column, and operands of mixed rank align by
  trailing-axis padding. Every other binder (circular convolution/correlation, shifting,
  segment shifting, matrix binding, VTB, context-dependent thinning) is applied per column
  internally, so a batched ``bind(A, B)`` returns one ``(D, N)``
  :class:`~pyhdc.Hypervector` without ``batch_dim``.
* Two-input :meth:`~pyhdc.Encoding.similarity` broadcasting over trailing axes: the
  result shape is the broadcast of the two operands' non-dimension axes. Two ``(D,)``
  vectors return a Python ``float``; every other pairing returns an array.
* First-class :meth:`~pyhdc.Encoding.permute`, :meth:`~pyhdc.Encoding.inverse`,
  :meth:`~pyhdc.Encoding.negative`, and :meth:`~pyhdc.Encoding.normalize` on
  :class:`~pyhdc.Encoding`, mirrored as methods on :class:`~pyhdc.Hypervector`
  (:meth:`~pyhdc.Hypervector.permute`, :meth:`~pyhdc.Hypervector.inverse`,
  :meth:`~pyhdc.Hypervector.negative`, :meth:`~pyhdc.Hypervector.normalize`). ``permute``
  is defined for every encoding (cyclic shift along axis 0); ``inverse`` / ``negative`` /
  ``normalize`` are wired per family and raise ``NotImplementedError`` where a family
  does not define them.
* Operator overloading on :class:`~pyhdc.Hypervector`: ``+`` (bundle), ``*`` (bind),
  ``/`` (unbind), ``~`` (inverse), ``>>`` (permute by ``+k``), ``<<`` (permute by ``-k``).
  A non-:class:`~pyhdc.Hypervector` operand to ``+ * /`` returns ``NotImplemented`` and
  Python raises ``TypeError``; a ``bool`` shift on ``>>`` / ``<<`` is rejected.
* Module-level :func:`~pyhdc.permute`, :func:`~pyhdc.inverse`, :func:`~pyhdc.negative`,
  :func:`~pyhdc.normalize`, and :func:`~pyhdc.unbind`, joining the existing
  :func:`~pyhdc.generate`, :func:`~pyhdc.zeros`, :func:`~pyhdc.bundle`,
  :func:`~pyhdc.bind`, and :func:`~pyhdc.stack`.
* :class:`~pyhdc.BSDC_THIN` is now reachable directly from the top level (previously only
  via ``pyhdc.encodings``); all 15 encodings are exported at the top level.

Changed (breaking)
~~~~~~~~~~~~~~~~~~~

* The misspelled ``BernoulliBiploar`` element generator is renamed to
  ``BernoulliBipolar``; the old name is removed. Any direct import of the old name in
  ``pyhdc.components.elements`` must be updated. The MAP_I, MAP_I_Bits, and MAP_B
  encodings that use it are unchanged in behavior.

**Migration guide**:

.. code-block:: python

   # The element generator was misspelled; import the corrected name.
   from pyhdc.components.elements import BernoulliBipolar   # was BernoulliBiploar

Changed
~~~~~~~

* Vectorized fast path for batched i.i.d. generation: with the default i.i.d. element
  generators (Bernoulli bipolar/binary, uniform bipolar/angles, normal real, Bernoulli
  sparse), ``generate(size=(D, N))`` draws the whole batch in one ``(D, *batch)`` call. It
  is reproducible under a fixed seed for a given batch shape, but no longer value-identical
  to generating the vectors one at a time. Dropping that cross-consistency removes a
  full-array transpose copy, about 10-24% faster than the prior order-preserving draw.
  Ordered and custom generators (and ``SparseSegmented`` for ``BSDC_SEG``) keep the
  per-vector loop and still match ``N`` successive single-vector ``generate`` calls.
* Non-batch-safe binders (circular convolution/correlation, shifting/segment-shifting for
  ``BSDC_S`` / ``BSDC_SEG`` / ``BSDC_THIN``, matrix binding for ``MBAT``, VTB, and
  context-dependent thinning for ``BSDC_CDT``) are applied per column when
  :meth:`~pyhdc.Encoding.bind` / ``unbind`` receives a batched (``ndim > 1``) input,
  returning one batched result. They previously produced a wrong result silently;
  single-vector inputs are unchanged.
* ``random_zone_count`` returns an ``int`` for a single ``(D,)`` result and an array for a
  batched result.
* ``ElementAdditionBits`` (MAP_I_Bits bundling) sums in a wide (int64) accumulator and
  clips the total once, saturating at the bounds. This replaces the previous per-addition
  clip, so results change when the running sum would have saturated mid-accumulation; it
  is vectorized and accepts a tuple of axes.
* ``DisjunctionThinned`` (BSDC_THIN bundling) thins a batched result without a per-column
  Python loop: each surviving column keeps a uniformly random ``ceil(D * density)``-subset
  of its set bits through a vectorized random-key selection.
* ``bundle(array, batch_dim=k)`` on a 3-D array reduces the other batch axis in one
  vectorized op instead of Python-looping the split slices (about 8x faster on a
  ``1000 x 20 x 500`` array). Ragged nested-list inputs, ``batch_dim=0``, and
  4-D-or-larger arrays keep the per-group path. For tie-randomizing bundlers the random
  values at tie coordinates now differ from the previous per-group draws (still random;
  ``batch_dim`` has no fixed-seed guarantee). ``axis=`` remains the preferred vectorized
  form, returning a single tensor instead of a list.

Deprecated
~~~~~~~~~~

* ``batch_dim`` on :meth:`~pyhdc.Encoding.bundle` / :meth:`~pyhdc.Encoding.bind` /
  :meth:`~pyhdc.Encoding.unbind` is deprecated and will be removed in a future release.
  Pass a batched array directly (operations batch automatically) or use ``axis=`` on
  ``bundle``. Passing ``batch_dim`` now emits a ``DeprecationWarning``.

----

v2.0.0: 2026-06-12
---------------------

Added
~~~~~

* Dimension-first ``(D, N)`` batched hypervectors. ``enc.generate(size=(D, N))``
  returns one :class:`~pyhdc.Hypervector` wrapping a ``(D, N)`` array whose columns
  are hypervectors. :meth:`~pyhdc.Encoding.bundle` collapses a ``(D, N)`` batch to a
  single ``(D,)`` prototype; :meth:`~pyhdc.Encoding.bind` / ``unbind`` operate per
  column.
* :meth:`~pyhdc.Hypervector.select`: select hypervectors (columns) from a ``(D, N)``
  batch by index, on both the NumPy and PyTorch backends.
* :func:`~pyhdc.stack`: backend-agnostic combine of hypervectors/batches into one
  ``(D, N)`` batch along the batch axis (a ``(D,)`` vector becomes a column).
* Global backend/device defaults: :func:`~pyhdc.prefer_torch`,
  :func:`~pyhdc.prefer_cuda`, :func:`~pyhdc.prefer_numpy`, :func:`~pyhdc.prefer_cpu`,
  :func:`~pyhdc.get_default_backend`, :func:`~pyhdc.get_default_device`. Encodings
  created without an explicit ``backend`` / ``device`` inherit these.
* Multi-mode similarity: a single ``(D, N)`` batch returns column 0 against each
  remaining column; two ``(D, N)`` batches return per-column pairs; a ``(D,)`` vector
  against a ``(D, N)`` batch broadcasts.
* :class:`~pyhdc.BSDC_THIN` is now exported at the top level.

Changed (breaking)
~~~~~~~~~~~~~~~~~~~

* Hypervector batches are now **dimension-first** ``(D, N)`` (each column is a
  hypervector), not batch-first ``(N, D)``. ``enc.generate(size=N)`` with an integer
  now returns a single ``N``-dimensional vector; use ``enc.generate(size=(D, N))``
  for a batch of ``N`` vectors.
* Batched :meth:`~pyhdc.Encoding.similarity` is column-wise over ``(D, N)`` instead
  of per-row over ``(N, D)``: ``similarity(A, B)`` returns per-column pairs, and
  ``similarity(batch)`` returns column 0 vs each remaining column.

**Migration guide**:

.. code-block:: python

   # A batch of N vectors was (N, D) in 1.1.0; make or transpose it to (D, N).
   batch = enc.generate(size=(10_000, 50))   # was enc.generate(size=50)

   # Batched similarity now indexes columns, not rows.
   sims   = enc.similarity(batch_a, batch_b)   # sims[i] = sim(batch_a[:, i], batch_b[:, i])
   member = batch[:, i]                         # was batch[i]

Fixed
~~~~~

* Batched generation is order-reproducible: ``generate(size=(D, N))`` yields the same
  vectors as ``N`` successive ``generate()`` calls under a fixed seed, and works for
  every generator (a 2-D ``size`` previously mis-ordered the columns or failed).

----

v1.1.0: 2026-05-24
---------------------

Added
~~~~~

* :class:`~pyhdc.BSDC_THIN` encoding: sparse binary with post-bundling
  random thinning to enforce a density constraint. Uses
  ``Shifting`` / ``InverseShifting`` for binding.
* ``DisjunctionThinned`` bundling function in ``pyhdc.components.bundling``:
  bitwise OR followed by random thinning to a target density.
* ``similarity_remap`` parameter on all encoding classes: optional callable
  applied to every similarity result before returning.
* ``remap_to_unit`` in ``pyhdc.components.similarity``: maps [-1, 1] → [0, 1].
  Works on scalars, NumPy arrays, and PyTorch tensors.
* PyTorch support for all four similarity functions (``CosineSimilarity``,
  ``HammingDistance``, ``Overlap``, ``AngleDistance``).
* Batched similarity calling conventions: ``(a, b)`` both 2-D returns
  per-row similarities; ``(arr,)`` single 2-D returns row 0 vs. rows 1+.

Changed (breaking)
~~~~~~~~~~~~~~~~~~~

* ``HammingDistance`` now returns **[-1, 1]** instead of [0, 1].
* ``Overlap`` now returns **[-1, 1]** instead of [0, 1].

**Migration guide**: any code comparing ``HammingDistance`` or ``Overlap``
output against thresholds in [0, 1] must be updated. The easiest fix:

.. code-block:: python

   from pyhdc.components.similarity import remap_to_unit

   # Option A: remap manually
   sim = hv1.similarity(hv2)
   sim_01 = remap_to_unit(sim)

   # Option B: remap automatically at the encoding level
   enc = pyhdc.BSC(dimension=10_000, similarity_remap=remap_to_unit)
   sim_01 = hv1.similarity(hv2)   # always in [0, 1]

Fixed
~~~~~

* ``MAP_I_Bits`` integer overflow on Python 3.9.
* All similarity functions now handle PyTorch tensors without falling back
  to NumPy.

----

v1.0.1: 2026-05-23
---------------------

Changed
~~~~~~~

* Added README.md with badges, installation instructions, and a quickstart
  example (omitted from the v1.0.0 tag; this patch ensures it appears on
  the PyPI release page).

----

v1.0.0: 2026-05-23
---------------------

Added
~~~~~

* Unit test suite covering all 14 encoding types, all 7 generator families,
  all components, and the hypervector API.
* Performance benchmark suite (``pytest-benchmark``).
* mypy static type checking configuration.
* Pre-commit hooks: autoflake, isort, black, pylint, mypy.
* ``CONTRIBUTING.md`` with developer setup and PR process.
* ``SECURITY.md`` with vulnerability reporting guidance.
* Codecov integration.
* TestPyPI and PyPI publish workflows with OIDC Trusted Publishing.

Fixed
~~~~~

* All internal imports changed from ``hdc.`` to ``pyhdc.`` namespace.
* ``DefaultGenerator._next_word`` integer overflow for ``word_size >= 32``.
* ``MBAT.bind`` incorrectly storing tuple as hypervector data.
* ``MAP_I_Bits`` wrong keyword argument names in ``ElementAdditionBits``.
* ``FeistelCounterGenerator`` non-deterministic round key generation.

----

v0.0.1: 2024-01-01
---------------------

Initial template release to PyPI.

Added
~~~~~

* Core encoding types: MAP_C, MAP_I, MAP_I_Bits, MAP_B, HRR, HRR_NoNorm,
  HRR_ConstNorm, FHRR, VTB, MBAT, BSC, BSDC_CDT, BSDC_S, BSDC_SEG
* Random number generator families: LCG, DLFSR, LFSR, LCA, PCG, Xorshift,
  ShiftedCounter
* Recovery algorithm framework (not yet public API)
* NumPy backend; PyTorch optional
* GitHub Actions CI: lint, test, PyPI publish workflows
