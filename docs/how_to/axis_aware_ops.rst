How to Work with (D, N, M) Batches
===================================

PyHDC is dimension-first: axis 0 is always the hypervector dimension ``D``, and
every trailing axis is batch structure. A single vector is ``(D,)``, a flat
batch is ``(D, N)``, and a two-level batch is ``(D, N, M)``. Each trailing-axis
slice is one hypervector, so ``batch[:, j]`` is column ``j`` and
``batch[:, :, k]`` is a ``(D, N)`` array.

This guide covers building a ``(D, N, M)`` tensor, reducing a chosen batch axis
with ``bundle``, computing similarity on a 3-D batch, and element-wise binding
that broadcasts a key against a batch. For the layout rules behind these
shapes, see :doc:`../user_manual/array_layout`.

Build a (D, N, M) tensor
------------------------

``generate`` takes a dimension-first ``size`` tuple. The first entry is ``D`` and
the remaining entries are batch axes. ``size=(D, N, M)`` returns a ``(D, N, M)``
tensor holding ``N * M`` hypervectors:

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_C(dimension=10_000)
   batch = enc.generate(size=(10_000, 4, 3))   # shape (10000, 4, 3)

   print(batch.shape)       # (10000, 4, 3)
   print(batch[:, 0, 0].shape)   # (10000,) one hypervector
   print(batch[:, :, 0].shape)   # (10000, 4) a (D, N) slab

Under a fixed seed, batched generation reproduces itself for a given shape. With
the i.i.d. element generators the whole batch is drawn in one call, so it is not
value-identical to ``N`` successive ``generate(size=D)`` calls. Ordered and
custom generators keep the per-vector loop and do match the sequential draws.
See :doc:`reproducibility` for the seeding details.

Bundle with the axis keyword
----------------------------

``bundle`` reduces one or more batch axes and returns a single
:class:`~pyhdc.Hypervector`. Axis 0 is the dimension and is never a legal reduce
axis, passing ``axis=0`` raises ``ValueError``. ``axis`` is the vectorized reduce
keyword, the older ``batch_dim`` keyword is deprecated (see Convention 4).

**Convention 1: default axis reduces the last batch axis**

With ``axis=None`` (the default), ``bundle`` collapses the last axis. A
``(D, N)`` batch collapses to ``(D,)``, a ``(D, N, M)`` tensor collapses to
``(D, N)``:

.. code-block:: python

   flat = enc.generate(size=(10_000, 50))   # (D, N)
   print(enc.bundle(flat).shape)            # (10000,)

   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)
   print(enc.bundle(cube).shape)              # (10000, 4)

**Convention 2: choose which batch axis to collapse**

Pass an integer ``axis`` to fold a specific batch axis. Reducing axis 1 of a
``(D, N, M)`` tensor leaves ``(D, M)``, reducing axis 2 leaves ``(D, N)``:

.. code-block:: python

   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)

   print(enc.bundle(cube, axis=1).shape)   # (10000, 3)
   print(enc.bundle(cube, axis=2).shape)   # (10000, 4)

Negative indices work and are normalized against the input rank, so
``axis=-1`` is the same as ``axis=2`` for a 3-D tensor.

**Convention 3: collapse several axes with a tuple**

The additive bundlers accept a tuple of axes and fold them together. Reducing
``(1, 2)`` of a ``(D, N, M)`` tensor collapses both batch axes to a single
``(D,)`` prototype:

.. code-block:: python

   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)
   print(enc.bundle(cube, axis=(1, 2)).shape)   # (10000,)

A tuple of axes applies to a single batched tensor. Bundling multiple separate
operands requires ``(D,)`` or ``(D, N)`` inputs and rejects any operand with
three or more dimensions. The tuple path is supported by the element-wise
additive bundlers (the MAP addition variants, the normalized and threshold
addition variants, ``AnglesOfElementAddition``, and the bitwise-OR disjunction
bundler, BSDC_S/SEG/CDT). BSDC_THIN (thinned OR) reduces a single axis only.
For the full per-operation list, see :doc:`../user_manual/bundling_operations`
and :doc:`bundle_hypervectors`.

**Convention 4: per-group bundles with axis**

To bundle each group of a 3-D batch on its own, reduce the *other* batch axis
with ``axis=`` and read the result columns. Reducing axis 1 of a ``(D, N, M)``
tensor bundles the ``N`` vectors at each ``M`` index and returns a single
``(D, M)`` tensor whose column ``j`` is the bundle of group ``j``:

.. code-block:: python

   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)

   groups = enc.bundle(cube, axis=1)   # (10000, 3): column j is group j
   print(groups.shape)        # (10000, 3)
   print(groups[:, 0].shape)  # (10000,)

The older ``batch_dim=`` keyword returned the same content as a Python list of
hypervectors. It is deprecated as of 2.1.0, emits a ``DeprecationWarning``, and
will be removed. Pass a batched array or use ``axis=`` instead. ``axis=`` also
keeps the fixed-seed reproducibility that the tie-randomizing bundlers (majority
vote, thinned OR) lose under ``batch_dim``.

Similarity on a 3-D batch needs an explicit axis
------------------------------------------------

``similarity`` reduces over axis 0 (the dimension). For a single input, the
``axis`` keyword selects which batch axis separates the query from the
candidates, ``axis`` is keyword-only.

**A (D, N) batch defaults to column 0 versus the rest**

A single ``(D, N)`` batch with no ``axis`` compares column 0 against columns 1
through ``N - 1``, returning ``N - 1`` scores:

.. code-block:: python

   batch = enc.generate(size=(10_000, 101))   # (D, N)
   sims  = enc.similarity(batch)              # shape (100,)
   # sims[i] = similarity(column 0, column i + 1)

This matches :ref:`the batched conventions <similarity-batched>` in
:doc:`compute_similarity`.

**A (D, N, M) batch requires you to name the axis**

A single batch with three or more dimensions has no default split axis. Calling
``similarity`` on it without ``axis`` raises
``ValueError("single-input similarity on a (D, N, M, ...) batch requires an
explicit axis")``. Name the batch axis to split:

.. code-block:: python

   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)

   # Wrong: no axis on a 3-D batch
   # enc.similarity(cube)   # ValueError

   # Right: split along axis 1 (head row vs the remaining rows)
   sims = enc.similarity(cube, axis=1)

The named axis is kept as a length-1 head against the length-(size minus one)
rest, so it broadcasts against the remaining batch axes. A single 1-D input is
rejected as single-input similarity needs at least a ``(D, N)`` batch.

Element-wise binding broadcasts
-------------------------------

Element-wise binders (MAP multiply, BSC XOR, FHRR angle add and subtract) align
operands by trailing-axis broadcasting, the same way NumPy and PyTorch do. A
``(D,)`` key binds against every column of a batch in one call:

.. code-block:: python

   enc   = pyhdc.MAP_C(dimension=10_000)
   key   = enc.generate()                    # (D,)
   batch = enc.generate(size=(10_000, 50))   # (D, N)

   bound = enc.bind(key, batch)   # (10000, 50): key bound to each column

Mixed ranks align by padding the lower-rank operand with trailing length-1
axes. A ``(D, N)`` operand binds against a ``(D, N, M)`` tensor by broadcasting
over the ``M`` axis:

.. code-block:: python

   keys = enc.generate(size=(10_000, 4))      # (D, N)
   cube = enc.generate(size=(10_000, 4, 3))   # (D, N, M)

   bound = enc.bind(keys, cube)   # (10000, 4, 3): keys broadcast over axis 2

Not every binder is element-wise. The convolution and correlation binders (the
HRR family), shifting and segment-shifting (the sparse families), matrix
binding (MBAT), VTB, and context-dependent thinning (BSDC_CDT) cannot broadcast
a per-coordinate rule across a batch. Pass a batch anyway and ``bind`` applies
the binder per column internally, returning one batched result. See
:doc:`bind_unbind` for the per-family binding details.

Putting it together
-------------------

The four shapes compose. The table below summarizes how each operation moves
between ranks:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Call
     - Input shape
     - Result shape
   * - ``generate(size=(D, N, M))``
     - ---
     - ``(D, N, M)``
   * - ``bundle(cube)`` (default axis)
     - ``(D, N, M)``
     - ``(D, N)``
   * - ``bundle(cube, axis=1)``
     - ``(D, N, M)``
     - ``(D, M)``
   * - ``bundle(cube, axis=(1, 2))``
     - ``(D, N, M)``
     - ``(D,)``
   * - ``similarity(cube, axis=1)``
     - ``(D, N, M)``
     - broadcast over the kept axes
   * - ``bind(key, batch)``
     - ``(D,)`` and ``(D, N)``
     - ``(D, N)``
   * - ``bind(keys, cube)``
     - ``(D, N)`` and ``(D, N, M)``
     - ``(D, N, M)``
