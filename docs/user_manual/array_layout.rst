Array Layout: Dimension-First (D, N, M)
=======================================

PyHDC stores every hypervector and every batch of hypervectors with the
hypervector dimension on **axis 0**. A single vector is ``(D,)``, a batch of
``N`` vectors is ``(D, N)``, and a tensor of ``N x M`` vectors is
``(D, N, M)``. The leading axis is always ``D``, every trailing axis is part
of the batch. This page explains the PyHDC convention, how the three primitives read
the axes, and why this ordering is the one that makes batched generation
reproducible.

PyHDC Convention
----------------

**Axis 0 is the dimension.** For any array you pass to PyHDC, ``array.shape[0]``
must equal the encoding's dimension ``D``. ``len(hv)`` equals ``hv.shape[0]``
equals ``D`` for every shape, whether the array holds one vector or a million.

**Trailing axes are the batch.** Axes 1 and beyond index *which* hypervector.
Each trailing-axis slice (each column) is one complete hypervector laid out
along axis 0. You index a batch the way you index any dimension-first array:

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)
   batch = enc.generate(size=(10_000, 8))   # (D, N) = (10000, 8)

   batch.shape          # (10000, 8)
   len(batch)           # 10000  -> this is D, not N
   batch[:, 0]          # (10000,) the first hypervector
   batch[:, :5]         # (10000, 5) the first five hypervectors
   batch[:, -1]         # (10000,) the last hypervector

**Axis 0 is never a legal reduce axis.** Reducing axis 0 would collapse the
coordinates of a single hypervector into a scalar, which is meaningless for
bundling. PyHDC enforces this: passing ``axis=0`` to :func:`bundle` raises
``ValueError("axis 0 is the hypervector dimension and cannot be reduced")``.
Negative axes are normalized first, so ``axis=-3`` on a 3D array resolves to
axis 0 and is rejected the same way.

Shape to meaning
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Shape
     - Name
     - Meaning
   * - ``(D,)``
     - single
     - One hypervector. Axis 0 is the dimension, there is no batch axis.
   * - ``(D, N)``
     - batch
     - ``N`` hypervectors. Column ``batch[:, j]`` is the ``j``-th vector.
   * - ``(D, N, M)``
     - tensor
     - ``N x M`` hypervectors. Slice ``tensor[:, i, j]`` is one vector;
       ``tensor[:, i]`` is the ``M``-vector row at index ``i``.
   * - ``(D, *batch)``
     - general
     - One hypervector per trailing-axis index, ``prod(batch)`` vectors total.

A lone ``(D,)`` input is promoted internally to ``(D, 1)`` when an operation
needs a batch axis, so a single vector behaves as a batch of one without you
reshaping it.

How bundle reads the axes
-------------------------

Bundling combines several hypervectors into one. It reduces a **batch** axis
(any trailing axis) and leaves axis 0 intact, because the output is itself a
hypervector of dimension ``D``.

With the default ``axis=None``, :func:`bundle` reduces the **last** axis:

.. code-block:: python

   batch = enc.generate(size=(10_000, 50))     # (D, N)
   summary = enc.bundle(batch)                 # reduces axis 1 -> (D,)

   tensor = enc.generate(size=(10_000, 4, 6))  # (D, N, M)
   rows = enc.bundle(tensor)                   # reduces axis 2 -> (D, 4)

Pass an explicit ``axis`` to choose a different batch axis, or a tuple of axes
to fold several at once:

.. code-block:: python

   tensor = enc.generate(size=(10_000, 4, 6))  # (D, N, M)
   enc.bundle(tensor, axis=1)                   # reduce N -> (D, 6)
   enc.bundle(tensor, axis=(1, 2))              # reduce N and M -> (D,)

A tuple of axes applies to a single batched tensor on the additive,
element-wise bundlers (the addition and OR variants). Bundling multiple
separate operands instead requires ``(D,)`` or ``(D, N)`` inputs. An operand
with three or more axes raises ``ValueError``. To bundle each group of a 3D
array on its own, reduce the *other* axis with ``axis=`` and read the result
columns.

The older ``batch_dim=`` keyword returned a Python list of results, it
is deprecated as of 2.1.0 and emits a ``DeprecationWarning``. See
:doc:`bundling_operations` for the per-bundler formulas.

How similarity reads the axes
-----------------------------

Similarity compares hypervectors, so it reduces **axis 0** (the dimension) and
broadcasts the trailing axes. The result shape is the broadcast of the two
operands' batch axes. Axis 0 disappears because each comparison produces one
scalar per vector pair.

.. code-block:: python

   key = enc.generate(size=10_000)             # (D,)
   codebook = enc.generate(size=(10_000, 200)) # (D, N)

   enc.similarity(key, codebook)               # (200,) one score per column
   enc.similarity(key, key)                    # Python float (both 1D)

A ``(D,)`` key broadcasts against every column of a ``(D, N)`` codebook, and a
``(D, N)`` batch broadcasts against a ``(D, N, M)`` tensor along axis 1. Two
1D inputs are the only case that returns a Python float, every other case
returns a numpy array or torch tensor whose shape is the broadcast of the
non-dimension axes.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - A shape
     - B shape
     - Result
   * - ``(D,)``
     - ``(D,)``
     - Python ``float``
   * - ``(D,)``
     - ``(D, N)``
     - ``(N,)``
   * - ``(D, N)``
     - ``(D, N)``
     - ``(N,)``
   * - ``(D,)``
     - ``(D, N, M)``
     - ``(N, M)``
   * - ``(D, N)``
     - ``(D, N, M)``
     - ``(N, M)`` (A broadcast over axis 2)
   * - ``(D, N, M)``
     - ``(D, N, M)``
     - ``(N, M)``

For the single-input form and the full set of modes, see
:ref:`batched calling conventions <similarity-batched>` and
:doc:`similarity_metrics`.

How element-wise bind reads the axes
------------------------------------

Element-wise binders (MAP multiply, BSC XOR, FHRR angle add/sub) operate per
coordinate along axis 0 and broadcast over the trailing batch axes, exactly
like similarity. A single ``(D,)`` key binds against every column of a batch:

.. code-block:: python

   key = enc.generate(size=10_000)             # (D,)
   values = enc.generate(size=(10_000, 32))    # (D, N)

   enc.bind(key, values)                       # (D, 32) key * each column

Mixed ranks align by trailing-axis padding, so a ``(D, N)`` batch binds
against a ``(D, N, M)`` tensor column for column. The non-element-wise binders
(circular convolution and correlation, shifting and segment shifting, matrix
binding, VTB, and CDT) cannot broadcast a per-coordinate rule. Pass a batch
anyway and ``bind`` applies the binder per column internally, returning one
batched result. The per-binder behavior is in :doc:`binding_operations`.

Why this layout
---------------

Putting ``D`` on axis 0 makes a batch a stack of columns: ``batch[:, j]`` is a
complete hypervector, bundling reduces a trailing axis, and axis 0 stays the
dimension throughout. Under a fixed seed and a fixed batch shape, ``generate``
reproduces itself.

With the i.i.d. element generators (Bernoulli, uniform, normal, sparse) the
whole ``(D, *batch)`` array is drawn in one vectorized call. That batch is
reproducible for a given seed and shape, but it is **not** value-identical to
generating the vectors one at a time: a single block draw and a per-vector loop
consume the random stream in different orders. Ordered generators (LCG, LFSR,
and the rest), any custom ``HDCGenerator``, and ``SparseSegmented`` keep the
per-vector loop, so for those a seeded batch equals ``N`` successive
single-vector calls.

For reproducible bundling, prefer ``axis=``. It reduces in place without the
extra random draw that the tie-randomizing bundlers (majority vote, thinned OR)
make at tie coordinates. The deprecated ``batch_dim`` bundling has no fixed-seed
guarantee.

The ``from_array`` convention
-----------------------------

When you wrap an existing array with :func:`Encoding.from_array`, the same
invariant applies and nothing is transposed or reshaped for you. Axis 0 must
equal ``self.dimension`` (it is ``D``), with the trailing axes as the batch:

.. code-block:: python

   import numpy as np

   data = np.random.choice([-1, 1], size=(10_000, 16))  # (D, N)
   hv = enc.from_array(data)                            # axis 0 is D

If you hand it an array with the batch on axis 0, every downstream operation
will read the wrong axis as the dimension. Build your arrays dimension-first
before wrapping them.
