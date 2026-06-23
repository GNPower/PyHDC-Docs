User Manual
===========

The User Manual explains *why* PyHDC works the way it does. It is
understanding-oriented: you can read it without running any code, and it
will deepen your intuition for HDC as a computational paradigm.

Contents
--------

.. toctree::
   :maxdepth: 2

   hdc_theory
   encodings_overview
   array_layout
   binding_operations
   bundling_operations
   unary_operations
   similarity_metrics
   generators
   backends
   components_overview

Overview
--------

:doc:`hdc_theory`
   The mathematics of hypervectors: near-orthogonality, capacity, and why
   the three primitives (bundle, bind, similarity) are sufficient for
   symbolic reasoning.

:doc:`encodings_overview`
   The ``Encoding`` base class and ``EncodingSpec`` design; a tour of all
   four encoding families (MAP, HRR/FHRR, Matrix, Binary/Sparse).

:doc:`array_layout`
   The dimension-first ``(D, N, M)`` convention: axis 0 is always the
   hypervector dimension, the trailing axes are the batch, and how bundling,
   similarity, and binding read each axis.

:doc:`binding_operations`
   Deep dive on every binding operation: element multiplication, circular
   convolution, XOR, shift, matrix transformation, and more.

:doc:`bundling_operations`
   Deep dive on every bundling operation: addition variants, normalisation,
   majority vote, bitwise OR, and thinned OR.

:doc:`unary_operations`
   The four single-vector operations (permute, inverse, negative, and
   normalize), which families define each, and the component behind it.

:doc:`similarity_metrics`
   Cosine, Hamming, Overlap, and Angle distance: formulas, output ranges,
   batched calling conventions, and remapping.

:doc:`generators`
   The seven generator families; LCG, LFSR, DLFSR, LCA, PCG, Xorshift,
   ShiftedCounter; with design rationale and a compatibility matrix.

:doc:`backends`
   How PyHDC's dual NumPy / PyTorch backend is implemented, when to use each,
   and how device placement works.

:doc:`components_overview`
   The ``pyhdc.components`` submodule: building blocks for custom encodings
   and advanced post-processing pipelines.
