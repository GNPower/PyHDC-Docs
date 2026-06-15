Tutorial 4: (Sparse) Binary Encodings (BSC and BSDC)
===================================================

Binary encodings represent hypervectors as arrays of 0s and 1s (or -1s and
+1s). Sparse variants keep most elements at 0. This tutorial covers the difference
between dense binary (BSC) and sparse binary (BSDC), the density saturation
problem with OR bundling and how BSDC_THIN solves it, and sequence encoding with
circular shifts.

**Prerequisites**: :doc:`tutorial_1_text_classification`

----

Dense binary: BSC
------------------

:class:`~pyhdc.BSC` (Binary Spatter Code) uses dense binary vectors where
each element is drawn from a Bernoulli distribution with p = 0.5: on average
half the elements are 1.

Binding is XOR (self-inverse), and similarity is Hamming distance remapped to
[-1, 1]:

.. code-block:: python

   import pyhdc
   import numpy as np

   enc = pyhdc.BSC(dimension=10_000)

   a = enc.generate()
   b = enc.generate()

   print(a.data.mean())          # ~= 0.5  # dense
   print(a.similarity(b))        # ~= 0.0  # unrelated
   print(a.similarity(a))        # 1.0     # identical

   # XOR binding is exactly self-inverse
   bound    = a.bind(b)
   recovered = bound.unbind(b)   # identical to a, not approximate
   print(np.allclose(recovered.data, a.data))   # True

----

Sparse binary: the BSDC family
--------------------------------

The BSDC family uses *sparse* binary vectors where only a small fraction
(typically 1-5%) of elements are 1. Sparsity improves orthogonality between
random vectors and mirrors the sparse activity of biological neurons.

PyHDC provides four BSDC variants:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Encoding
     - Distinguishing feature
   * - ``BSDC_CDT``
     - Context-dependent thinning during bundling; unbind **not** supported
   * - ``BSDC_S``
     - Binding via circular shift; unbind supported
   * - ``BSDC_SEG``
     - Per-segment circular shift; useful for positional encodings
   * - ``BSDC_THIN``
     - Random thinning after OR-bundle to maintain target density (v1.1.0+)

.. code-block:: python

   enc_s = pyhdc.BSDC_S(dimension=10_000)
   v     = enc_s.generate()
   print(v.data.mean())   # ~= 0.01–0.05  # sparse

----

The density growth problem
---------------------------

BSDC encodings use bitwise OR for bundling. OR is a natural set-union
operation for sparse binary vectors, but it has a fatal flaw: each OR
operation can only turn bits *on*, never off. After many bundles, density
creeps toward 1.0 and all vectors look the same.

.. code-block:: python

   enc_s  = pyhdc.BSDC_S(dimension=10_000)
   result = enc_s.generate()
   print(f"Step  0: density = {result.data.mean():.4f}")

   for step in range(1, 21):
       result = result.bundle(enc_s.generate())
       if step % 4 == 0:
           print(f"Step {step:2d}: density = {result.data.mean():.4f}")

Expected output:

.. code-block:: text

   Step  0: density = 0.0106
   Step  4: density = 0.0507
   Step  8: density = 0.0856
   Step 12: density = 0.1205
   Step 16: density = 0.1518
   Step 20: density = 0.1847

If you continued to step 200, density would approach 1.0 and every
hypervector would be indistinguishable.

----

Solving density growth: BSDC_THIN
-----------------------------------

:class:`~pyhdc.BSDC_THIN` applies random thinning after each OR-bundle step.
Thinning randomly clears bits until the vector reaches a target density,
keeping density bounded regardless of how many bundles you perform.

.. code-block:: python

   enc_thin = pyhdc.BSDC_THIN(dimension=10_000, density=0.01)   # default density = 0.5
   result   = enc_thin.generate()
   print(f"Step  0: density = {result.data.mean():.4f}")

   for step in range(1, 21):
       result = result.bundle(enc_thin.generate())
       if step % 4 == 0:
           print(f"Step {step:2d}: density = {result.data.mean():.4f}")

Expected output:

.. code-block:: text

   Step  0: density = 0.0111
   Step  4: density = 0.0100
   Step  8: density = 0.0100
   Step 12: density = 0.0100
   Step 16: density = 0.0100
   Step 20: density = 0.0100

Density stays near the initial target throughout.

You can control the target density explicitly:

.. code-block:: python

   enc_dense  = pyhdc.BSDC_THIN(dimension=10_000, density=0.01)
   # density is determined by BernoulliSparse element generator
   # To change: pass a custom element generator, or use the density parameter if available

----

Sequence encoding with BSDC_S
-------------------------------

:class:`~pyhdc.BSDC_S` binds by performing a *circular shift* of the
hypervector by one position. Binding the *k*-th element with a shift of *k*
positions encodes position. This makes it natural for sequence encoding:

.. code-block:: python

   enc_s = pyhdc.BSDC_S(dimension=10_000)

   # Character hypervectors
   chars = {c: enc_s.generate() for c in 'abcdefghijklmnopqrstuvwxyz'}

   def encode_sequence(seq):
       """Encode a sequence by binding each element to its shifted position."""
       hvs = []
       hv  = chars[seq[0]]                # position 0: no shift
       hvs.append(hv)
       for ch in seq[1:]:
           hv = chars[ch].bind(hv)        # each bind shifts the previous result
           hvs.append(hv)
       return pyhdc.bundle(*hvs)

   cat = encode_sequence('cat')
   bat = encode_sequence('bat')
   rat = encode_sequence('rat')
   car = encode_sequence('car')

   print(cat.similarity(cat))   # 1.0
   print(cat.similarity(bat))   # ~= low  # different first character
   print(cat.similarity(car))   # ~= moderate: share 'c' and 'a'

The circular shift means the same character at different positions maps to
different hypervectors, preserving order information.

----

Similarity range and remapping
-----------------------------------------------

As of v1.1.0, all similarity functions in PyHDC return values in **[-1, 1]**:

* ``-1`` : maximally dissimilar (all bits different for Hamming; zero overlap)
* ``0`` : unrelated (expected for random pairs)
* ``+1`` : identical

In v1.0.x, ``HammingDistance`` and ``Overlap`` returned [0, 1]. If you are
migrating from v1.0.x, use ``similarity_remap`` to restore the old behaviour:

.. code-block:: python

   from pyhdc.components.similarity import remap_to_unit

   # Remap [-1, 1] -> [0, 1]
   enc_remap = pyhdc.BSC(dimension=10_000, similarity_remap=remap_to_unit)

   a = enc_remap.generate()
   print(a.similarity(a))   # 1.0  (was 1.0 in v1.0.x: unchanged)
   print(a.similarity(enc_remap.generate()))   # ~= 0.5  (was ~= 0.5 in v1.0.x)

You can also apply ``remap_to_unit`` manually to any similarity result:

.. code-block:: python

   from pyhdc.components.similarity import remap_to_unit, HammingDistance

   enc  = pyhdc.BSC(dimension=10_000)
   a, b = enc.generate(), enc.generate()

   raw     = a.similarity(b)              # in [-1, 1]
   remapped = remap_to_unit(raw)          # in  [0, 1]
   print(raw, remapped)

----

Choosing density
-----------------

Lower density means greater orthogonality between random vectors, which
means higher capacity. However, very low density (< 0.001) requires very
large dimensions to give enough 1s per vector for stable operations.

Practical guidelines:

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Density
     - Dimension
     - Notes
   * - 0.1-0.5
     - Any
     - Dense binary (BSC territory); moderate orthogonality
   * - 0.01-0.05
     - ≥ 1,000
     - Standard BSDC range; good balance of sparsity and stability
   * - 0.001-0.01
     - ≥ 10,000
     - Very sparse; very high capacity; needs large D for stability

----

Summary
-------

In this tutorial you:

* Compared dense binary (BSC) and sparse binary (BSDC) encodings
* Demonstrated the density growth problem with repeated OR-bundling
* Fixed density growth using ``BSDC_THIN``
* Encoded a character sequence using circular-shift binding with ``BSDC_S``
* Understood the v1.1.0 similarity range change and how to use ``remap_to_unit``

----

What's next
-----------

* :doc:`tutorial_5_custom_generators` : seeded, reproducible experiments
* :doc:`../how_to/control_density` : practical density control recipes
* :doc:`../user_manual/encodings_overview` : full encoding family comparison
