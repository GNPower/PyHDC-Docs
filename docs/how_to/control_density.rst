How to Control Density in Sparse Binary Encodings
===================================================

Sparse binary encodings (BSDC family) work best when the fraction of 1-bits
(*density*) stays well below 0.5. The sections below show how to measure
density and keep it bounded during bundling.

Measuring density
-----------------

Density is simply the mean of a binary hypervector's data array:

.. code-block:: python

   import pyhdc

   enc = pyhdc.BSDC_S(dimension=10_000)
   hv  = enc.generate()
   print(f"density = {hv.data.mean():.4f}")   # ~= 0.01

The density growth problem
---------------------------

BSDC uses bitwise OR for bundling. OR can only turn bits *on*, so repeated
bundling drives density toward 1.0:

.. code-block:: python

   enc    = pyhdc.BSDC_S(dimension=10_000)
   result = enc.generate()
   print(f"step  0: density = {result.data.mean():.4f}")

   for i in range(1, 21):
       result = result.bundle(enc.generate())
       if i % 5 == 0:
           print(f"step {i:2d}: density = {result.data.mean():.4f}")

   # density increases with each step

Solving it with BSDC_THIN
---------------------------

:class:`~pyhdc.BSDC_THIN` applies random thinning after each OR step; bits
are randomly cleared until the density returns to the initial level:

.. code-block:: python

   enc    = pyhdc.BSDC_THIN(dimension=10_000)
   result = enc.generate()
   print(f"step  0: density = {result.data.mean():.4f}")

   for i in range(1, 21):
       result = result.bundle(enc.generate())
       if i % 5 == 0:
           print(f"step {i:2d}: density = {result.data.mean():.4f}")

   # density stays stable throughout

Using DisjunctionThinned directly
-----------------------------------

If you are building a custom pipeline with the components submodule, you can
access the thinned OR operation directly:

.. code-block:: python

   from pyhdc.components.bundling import DisjunctionThinned
   import numpy as np

   a = enc.generate().data
   b = enc.generate().data

   result = DisjunctionThinned(a, b, density=0.05)

Density guidelines
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Density
     - Dimension
     - Notes
   * - ≥ 0.1
     - Any
     - Dense binary; more like BSC; less sparse-binary advantage
   * - 0.01–0.05
     - ≥ 1,000
     - Recommended BSDC range; good balance of capacity and stability
   * - ≤ 0.01
     - ≥ 10,000
     - Very sparse; high capacity; need large D for enough 1s per vector

Lower density means greater orthogonality between random vectors and higher
theoretical capacity, but too-low density at small dimensions means each
vector has very few 1s, making similarity estimates noisy.
