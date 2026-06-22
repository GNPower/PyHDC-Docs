How to Bundle Hypervectors
===========================

Bundling combines multiple hypervectors into one that is *similar to all
inputs*. Four equivalent ways to bundle are provided.

Four bundling methods
---------------------

**1. Instance method**: bundle one hypervector with others:

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)
   a, b, c = enc.generate(), enc.generate(), enc.generate()

   result = a.bundle(b, c)   # result is similar to a, b, and c

**2. Encoding method**: bundle via the encoding object:

.. code-block:: python

   result = enc.bundle(a, b, c)

You can also pass a list:

.. code-block:: python

   hvs    = [enc.generate() for _ in range(10)]
   result = enc.bundle(*hvs)

**3. Convenience function**: module-level shortcut:

.. code-block:: python

   result = pyhdc.bundle(a, b, c)

**4. Batched bundling**: bundle multiple groups at once:

.. code-block:: python

   # Bundle [[a,b], [c,d]] -> returns [bundle(a,b), bundle(c,d)]
   results = enc.bundle([a, b], [c, d])   # list of two Hypervectors

Collapse a whole batch to one prototype
----------------------------------------

A batch of ``N`` hypervectors is a ``(D, N)`` array, where each column is one
hypervector. Passing such a batch to ``bundle`` folds over the ``N`` columns and
returns a single ``(D,)`` prototype:

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_C(dimension=10_000)
   batch = enc.generate(size=(10_000, 50))   # 50 vectors as columns (shape: (10000, 50))
   # collapse the 50 columns into one prototype
   result = enc.bundle(batch)
   print(result.shape)   # (10000,)

Choose which axis to collapse
------------------------------

A higher-rank batch is a ``(D, N, M)`` tensor, where axis 0 is the dimension
``D`` and each trailing-axis column is one hypervector. The ``axis`` keyword
selects which batch axis ``bundle`` folds over.

**Default is the last batch axis.** With ``axis=None`` (the default), ``bundle``
reduces the last axis. A ``(D, N)`` batch still collapses to ``(D,)``, and a
``(D, N, M)`` tensor collapses to ``(D, N)``:

.. code-block:: python

   import pyhdc

   enc    = pyhdc.MAP_C(dimension=10_000)
   tensor = enc.generate(size=(10_000, 8, 5))   # shape: (10000, 8, 5)

   # default axis=None reduces the last axis (axis 2)
   result = enc.bundle(tensor)
   print(result.shape)   # (10000, 8)

**Pass an explicit axis** to collapse a different batch axis. Axis 0 is the
hypervector dimension and is never reducible, passing ``axis=0`` raises
``ValueError``. Negative indices are normalized the usual way:

.. code-block:: python

   # reduce axis 1, keeping axis 2 as the remaining batch
   result = enc.bundle(tensor, axis=1)
   print(result.shape)   # (10000, 5)

**A tuple of axes** folds several batch axes at once. This applies to the
additive, element-wise bundlers (the MAP, HRR, FHRR addition variants and the
BSDC bitwise-OR disjunction, BSDC_S/SEG/CDT), and it operates on a single
batched tensor. BSDC_THIN (thinned OR) reduces a single axis only:

.. code-block:: python

   # reduce axes 1 and 2 together -> one prototype
   result = enc.bundle(tensor, axis=(1, 2))
   print(result.shape)   # (10000,)

Get per-group results with axis
--------------------------------

``axis`` reduces the selected batch axis in place and returns a **single**
``Hypervector``. To bundle each group of a 3-D batch separately, reduce the
*other* batch axis and read the result columns, column ``j`` is group ``j``'s
bundle:

.. code-block:: python

   single = enc.bundle(tensor, axis=2)   # one Hypervector, shape (10000, 8)
   groups = enc.bundle(tensor, axis=1)   # one Hypervector, shape (10000, 5)
   #                                       groups[:, j] is the bundle of group j

The older ``batch_dim=`` keyword split a 3-D array along a dimension and returned
a Python list of hypervectors. It is deprecated as of 2.1.0, emits a
``DeprecationWarning``, and will be removed. ``axis=`` returns the same content as
one tensor. Passing both ``axis`` and ``batch_dim`` raises ``ValueError``.

Zero vector as the bundle identity
------------------------------------

The zero hypervector acts as the additive identity for bundling; bundling
anything with zero leaves it unchanged:

.. code-block:: python

   zero   = enc.zeros()
   result = enc.bundle(a, zero)
   print(result.similarity(a))   # ~= 1.0

This is useful when building bundles iteratively:

.. code-block:: python

   accumulator = enc.zeros()
   for hv in hvs:
       accumulator = enc.bundle(accumulator, hv)

Capacity limits
----------------

Bundling is lossy: each bundle adds noise to every component. The more
hypervectors you bundle, the harder it is to distinguish individual members
via similarity.

Approximate rule of thumb: bundling more than :math:`O(N \times ln(M))` vectors
into a single hypervector of dimension :math:`D` causes the similarity to each
component to drop below a useful threshold.

Encoding-specific notes
------------------------

Different encoding families use different bundling implementations, but the
interface is always the same:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Encoding
     - Bundling behaviour
   * - MAP_C
     - Element-wise addition then clip to [-1,1]; ties broken randomly (``random_choice_range`` parameter)
   * - MAP_I
     - Element-wise addition (plain sum), near-zero-sum ties broken randomly (``random_choice_range``)
   * - MAP_B
     - Element-wise addition then sign threshold (majority vote) to {-1, +1}
   * - HRR
     - Element-wise addition followed by L2 normalisation
   * - HRR_NoNorm
     - Element-wise addition without normalisation (vectors grow in magnitude)
   * - FHRR
     - Sum phasors, extract resultant angle
   * - BSC
     - Majority-vote threshold: each element is 1 if more than half the inputs are 1
   * - BSDC family
     - Bitwise OR. BSDC_THIN applies random thinning after OR to maintain density
