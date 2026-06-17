How to Compute Similarity
==========================

Similarity measures how related two hypervectors are. PyHDC returns values
in **[-1, 1]** (1 = identical, 0 = unrelated, -1 = maximally dissimilar).

Basic usage
-----------

**Instance method** (most common):

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)
   a   = enc.generate()
   b   = enc.generate()

   sim = a.similarity(b)   # float

**Encoding method** (same result):

.. code-block:: python

   sim = enc.similarity(a, b)

.. _similarity-batched:

Batched similarity: calling conventions
---------------------------------------

Hypervectors are dimension-first: a single vector has shape ``(D,)`` and a batch
of ``N`` vectors has shape ``(D, N)`` (each **column** is a hypervector). Both
``Hypervector.similarity()`` and ``Encoding.similarity()`` reduce over axis 0
(the dimension) and support these shapes:

**Convention 1; two 1-D vectors -> scalar**

.. code-block:: python

   a = enc.generate()   # shape (10000,)
   b = enc.generate()   # shape (10000,)
   sim = enc.similarity(a, b)   # float

**Convention 2; two (D, N) batches -> 1-D array (per-column pairs)**

Element ``i`` of the result is ``similarity(A[:, i], B[:, i])``:

.. code-block:: python

   batch_a = enc.generate(size=(10_000, 50))   # shape (10000, 50)
   batch_b = enc.generate(size=(10_000, 50))   # shape (10000, 50)
   sims    = enc.similarity(batch_a, batch_b)   # shape (50,)

**Convention 3; single (D, N) batch -> 1-D array (column 0 vs the rest)**

Column 0 is the query; columns 1+ are the candidates:

.. code-block:: python

   query_plus_codebook = enc.generate(size=(10_000, 101))   # shape (10000, 101)
   sims = enc.similarity(query_plus_codebook)                # shape (100,)
   # sims[i] = similarity(column 0, column i+1)

**Convention 4; one vector vs a batch -> 1-D array (broadcast)**

A ``(D,)`` vector compared against every column of a ``(D, N)`` batch:

.. code-block:: python

   query    = enc.generate()                    # shape (10000,)
   codebook = enc.generate(size=(10_000, 100))   # shape (10000, 100)
   sims     = enc.similarity(query, codebook)    # shape (100,)

**Batched list form at the encoding level**

You can also pass two equal-length lists of ``Hypervector`` objects:

.. code-block:: python

   hvs_a = [enc.generate() for _ in range(5)]
   hvs_b = [enc.generate() for _ in range(5)]
   sims  = enc.similarity(hvs_a, hvs_b)   # list of 5 floats

Output ranges by encoding
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Encoding family
     - Similarity metric
     - Output range
   * - MAP_C, MAP_I, MAP_I_Bits, MAP_B
     - Cosine
     - [-1, 1]
   * - HRR, HRR_NoNorm, HRR_ConstNorm
     - Cosine
     - [-1, 1]
   * - FHRR
     - Angle distance
     - [-1, 1]
   * - VTB, MBAT
     - Cosine
     - [-1, 1]
   * - BSC
     - Hamming (remapped)
     - [-1, 1]  *(was [0,1] in v1.0.x)*
   * - BSDC family
     - Overlap (remapped)
     - [-1, 1]  *(was [0,1] in v1.0.x)*

Remapping to [0, 1]
--------------------

If your downstream code expects [0, 1] (e.g., scikit-learn metrics), use
``similarity_remap`` on the encoding constructor:

.. code-block:: python

   from pyhdc.components.similarity import remap_to_unit

   enc = pyhdc.BSC(dimension=10_000, similarity_remap=remap_to_unit)
   a   = enc.generate()
   print(a.similarity(a))   # 1.0
   print(a.similarity(enc.generate()))   # ~= 0.5

Or apply ``remap_to_unit`` manually:

.. code-block:: python

   raw      = a.similarity(b)          # in [-1, 1]
   remapped = remap_to_unit(raw)        # in [0, 1]

Nearest-neighbour lookup
-------------------------

Find the closest match to a query in a small codebook:

.. code-block:: python

   codebook = {name: enc.generate() for name in ['red','green','blue','yellow']}
   query    = codebook['red'].bundle(enc.generate())   # noisy version of red

   best = max(codebook, key=lambda k: query.similarity(codebook[k]))
   print(best)   # red

For large codebooks (thousands of items), keep the codebook as one ``(D, N)``
batch and compare in a single vectorized call:

.. code-block:: python

   import numpy as np

   enc      = pyhdc.MAP_C(dimension=10_000)
   codebook = enc.generate(size=(10_000, 5_000))   # (D, N): 5000 items as columns
   query    = enc.generate()                        # (D,)

   sims     = enc.similarity(query, codebook)        # shape (5000,) broadcast
   best_idx = int(np.argmax(sims))
   print(best_idx)

   # Or stack the query as column 0 and use convention 3:
   stacked  = pyhdc.stack([query, codebook])          # (D, 5001)
   sims     = enc.similarity(stacked)                 # shape (5000,)

   # Pull specific candidates back out of the batch with select:
   top3     = codebook.select(np.argsort(sims)[-3:])  # (D, 3)
