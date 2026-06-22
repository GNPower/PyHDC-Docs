Tutorials
=========

The tutorials walk you through complete, runnable examples that demonstrate
how to use PyHDC to solve real problems. Each tutorial introduces new concepts
progressively, so reading them in order is recommended for beginners.

**Recommended order**

.. list-table::
   :header-rows: 1
   :widths: 5 35 20 40

   * - #
     - Title
     - Audience
     - What you will learn
   * - 1
     - :doc:`tutorial_1_text_classification`
     - Beginner
     - Codebooks, n-gram encoding, class prototypes, similarity-based classification
   * - 2
     - :doc:`tutorial_2_associative_memory`
     - Beginner-Intermediate
     - Key-value binding, record construction, noisy query, capacity limits
   * - 3
     - :doc:`tutorial_3_pytorch_gpu`
     - Intermediate
     - PyTorch backend, GPU encoding, batched operations, performance profiling
   * - 4
     - :doc:`tutorial_4_sparse_binary`
     - Intermediate
     - BSC, BSDC density control, BSDC_THIN, sequence encoding, similarity remapping
   * - 5
     - :doc:`tutorial_5_custom_encodings`
     - Intermediate-Advanced
     - Subclassing Encoding, EncodingSpec, wiring components, custom inverse / permute / normalize
   * - 6
     - :doc:`tutorial_6_custom_generators`
     - Intermediate-Advanced
     - Seeded generators, generator families, reproducibility, custom generator subclass

Tutorials 3-6 are independent of each other: you can read them in any order
after completing Tutorials 1 and 2.

.. toctree::
   :hidden:

   tutorial_1_text_classification
   tutorial_2_associative_memory
   tutorial_3_pytorch_gpu
   tutorial_4_sparse_binary
   tutorial_5_custom_encodings
   tutorial_6_custom_generators
