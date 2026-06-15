How-To Guides
=============

How-to guides solve specific, practical problems. They assume you already know
the basics. If you are new to PyHDC, start with the
:doc:`../getting_started/quickstart` and :doc:`../tutorials/index` first.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Guide
     - Problem solved
   * - :doc:`choose_encoding`
     - Which encoding should I use for my task?
   * - :doc:`bundle_hypervectors`
     - How do I bundle multiple hypervectors?
   * - :doc:`bind_unbind`
     - How do I store and retrieve key-value pairs?
   * - :doc:`compute_similarity`
     - How do I compare hypervectors, including in batch?
   * - :doc:`switch_backends`
     - How do I move between NumPy and PyTorch / GPU?
   * - :doc:`reproducibility`
     - How do I make my experiments reproducible?
   * - :doc:`similarity_remap`
     - How do I remap similarity output to [0, 1]?
   * - :doc:`control_density`
     - How do I keep sparse binary vectors from becoming dense?
   * - :doc:`handle_exceptions`
     - How do I handle PyHDC errors gracefully?
   * - :doc:`wrap_arrays`
     - How do I wrap an existing NumPy array as a Hypervector?

.. toctree::
   :hidden:

   choose_encoding
   bundle_hypervectors
   bind_unbind
   compute_similarity
   switch_backends
   reproducibility
   similarity_remap
   control_density
   handle_exceptions
   wrap_arrays
