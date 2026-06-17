How to Remap Similarity Output to [0, 1]
==========================================

Since v1.1.0, all PyHDC similarity functions return values in **[-1, 1]**.
If your code expects [0, 1], for example when feeding results to scikit-learn
metrics or comparing against v1.0.x thresholds, use ``remap_to_unit``.

The remap function
-------------------

``remap_to_unit`` applies the linear transformation:

.. math::

   \text{remapped} = \frac{\text{sim} + 1}{2}

This maps −1 → 0, 0 → 0.5, and +1 → 1. It works on scalars, NumPy arrays,
and PyTorch tensors.

.. code-block:: python

   from pyhdc.components.similarity import remap_to_unit

   print(remap_to_unit(-1.0))    # 0.0
   print(remap_to_unit(0.0))     # 0.5
   print(remap_to_unit(1.0))     # 1.0

Apply it automatically via the encoding
-----------------------------------------

Pass ``similarity_remap=remap_to_unit`` to the encoding constructor. Every
similarity call on that encoding is automatically remapped:

.. code-block:: python

   import pyhdc
   from pyhdc.components.similarity import remap_to_unit

   enc = pyhdc.BSC(dimension=10_000, similarity_remap=remap_to_unit)
   a   = enc.generate()
   b   = enc.generate()

   print(a.similarity(a))   # 1.0
   print(a.similarity(b))   # ~= 0.5  (unrelated; 0.5 in [0,1] means orthogonal)

Apply it manually
------------------

If you only need to remap some calls, apply it after the fact:

.. code-block:: python

   enc = pyhdc.BSC(dimension=10_000)   # default: no remap
   raw      = a.similarity(b)          # [-1, 1]
   remapped = remap_to_unit(raw)        # [0, 1]

Custom remap function
----------------------

The ``similarity_remap`` parameter accepts any callable that maps a scalar or
array to a scalar or array of the same shape. Examples:

.. code-block:: python

   import numpy as np

   # Sigmoid remap: maps all [-1, 1] to (0, 1) with smooth S-curve
   def sigmoid_remap(x):
       return 1 / (1 + np.exp(-5 * x))   # scale factor 5 sharpens the transition

   enc = pyhdc.MAP_C(dimension=10_000, similarity_remap=sigmoid_remap)

   # Hard threshold: returns 1 if similar, 0 if not
   def threshold_remap(x):
       return (x > 0.3).astype(float)

Batched similarity with remap
-------------------------------

Remapping is applied element-wise, so it works correctly with all batched
calling conventions:

.. code-block:: python

   enc   = pyhdc.MAP_C(dimension=10_000, similarity_remap=remap_to_unit)
   batch = enc.generate(size=(10_000, 10))   # (D, N): each column is a hypervector
   query = enc.generate()

   # Per-column pairs: result[i] = similarity(batch[:, i], other[:, i])
   sims = enc.similarity(batch, enc.generate(size=(10_000, 10)))   # shape (10,), all in [0,1]

Migration from v1.0.x
----------------------

In v1.0.x, ``HammingDistance`` and ``Overlap`` (used by BSC and BSDC) returned
[0, 1]. In v1.1.0 they return [-1, 1]. To restore the old behaviour globally,
add ``similarity_remap=remap_to_unit`` when constructing any BSC or BSDC
encoding.
