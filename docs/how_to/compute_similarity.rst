How to Compute Similarity
==========================

Similarity measures how related two hypervectors are. PyHDC returns values
in **[-1, 1]** (1 = identical, 0 = unrelated, −1 = maximally dissimilar).

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

Batched similarity: three calling conventions
----------------------------------------------

As of v1.1.0, both ``Hypervector.similarity()`` and ``Encoding.similarity()``
support three input shapes:

**Convention 1; two 1-D vectors → scalar**

.. code-block:: python

   a = enc.generate()   # shape (10000,)
   b = enc.generate()   # shape (10000,)
   sim = enc.similarity(a, b)   # float

**Convention 2; two 2-D batches → 1-D array (per-row pairs)**

Row ``i`` of the result is ``similarity(a[i], b[i])``:

.. code-block:: python

   batch_a = enc.generate(size=50)   # shape (50, 10000)
   batch_b = enc.generate(size=50)   # shape (50, 10000)
   sims    = enc.similarity(batch_a, batch_b)   # shape (50,)

**Convention 3; single 2-D batch → 1-D array (first row vs. rest)**

Row 0 is the query; rows 1+ are the candidates:

.. code-block:: python

   query_plus_codebook = enc.generate(size=101)   # shape (101, 10000)
   sims = enc.similarity(query_plus_codebook)      # shape (100,)
   # sims[i] = similarity(query_plus_codebook[0], query_plus_codebook[i+1])

**Batched list form at the encoding level**

You can also pass lists of ``Hypervector`` objects:

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
   print(a.similarity(enc.generate()))   # ≈ 0.5

Or apply ``remap_to_unit`` manually:

.. code-block:: python

   raw      = a.similarity(b)          # in [-1, 1]
   remapped = remap_to_unit(raw)        # in [0, 1]

Nearest-neighbour lookup
-------------------------

Find the closest match to a query in a codebook:

.. code-block:: python

   codebook = {name: enc.generate() for name in ['red','green','blue','yellow']}
   query    = codebook['red'].bundle(enc.generate())   # noisy version of red

   best = max(codebook, key=lambda k: query.similarity(codebook[k]))
   print(best)   # red

For large codebooks (thousands of items), use batched convention 3 for speed:

.. code-block:: python

   import numpy as np

   names = list(codebook)
   hvs   = [codebook[n] for n in names]

   # Stack query + codebook into one batch
   enc_torch = pyhdc.MAP_C(dimension=10_000, backend="torch")
   stacked   = enc_torch.generate(size=len(hvs) + 1)
   # (in practice: assign query to stacked[0] and codebook to stacked[1:])
   sims      = enc_torch.similarity(stacked)   # shape (len(hvs),)
   best_idx  = int(np.argmax(sims))
   print(names[best_idx])
