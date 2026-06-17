Changelog
=========

All notable changes to PyHDC are documented here. The project follows
`Semantic Versioning <https://semver.org/>`_ and
`Keep a Changelog <https://keepachangelog.com/>`_ conventions.

The source is `CHANGELOG.md on GitHub
<https://github.com/GNPower/PyHDC/blob/main/CHANGELOG.md>`_.

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
