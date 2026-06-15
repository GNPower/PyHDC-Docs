Generation Module
=================

.. module:: pyhdc.generation

The ``pyhdc.generation`` module provides deterministic random number
generators for reproducible hypervector generation.

HDCGenerator (abstract base)
------------------------------

.. autoclass:: pyhdc.generation.base.HDCGenerator
   :members:
   :show-inheritance:

**Constructor**:

.. code-block:: python

   HDCGenerator(seed=None)

:param seed: Optional integer seed. If ``None``, the generator is initialised
             from a random source.

**Abstract methods** (must be implemented by subclasses):

* ``_configure_internal()``: called during ``__init__``; sets up state from parameters
* ``_next_bit()``: return the next bit (0 or 1); raise ``NotImplementedError`` if not supported
* ``_next_word(word_size=32)``: return the next integer of ``word_size`` bits
* ``set_parameters(**kwargs)``: update generator-specific parameters
* ``get_parameters()``: return current parameter dict
* ``reset()``: restore to initial state

**Concrete methods**:

* ``generate_bits(length)`` → ``list[int]``
* ``generate_words(length, word_size=32)`` → ``list[int]``
* ``generate_floats(length, min_val=-1.0, max_val=1.0)`` → ``list[float]``
* ``supports_bits()`` → ``bool``
* ``supports_words()`` → ``bool``
* ``supports_floats()`` → ``bool``
* ``get_state()`` → internal state (type is generator-dependent)
* ``set_seed(seed)``: set a new seed and reinitialise

DefaultGenerator
-----------------

.. autoclass:: pyhdc.generation.base.DefaultGenerator
   :show-inheritance:

NumPy-backed generator used when no custom generator is specified. Wraps
``numpy.random.default_rng()`` internally.

:param seed: Optional integer seed passed to ``numpy.random.default_rng``.

LCG family
----------

.. autoclass:: pyhdc.generation.lcg.LCGGenerator
   :show-inheritance:

Linear Congruential Generator: :math:`X_{n+1} = (a X_n + c) \bmod m`

:param modulus: ``m`` (default: 2\ :sup:`32`)
:param multiplier: ``a`` (default: 1664525)
:param increment: ``c`` (default: 1013904223)
:param seed: Initial state

.. autoclass:: pyhdc.generation.lcg.MultiplicativeLCGGenerator
   :show-inheritance:

LCG with ``c = 0`` (multiplicative / Lehmer generator).

.. py:class:: CommonLCGGenerators

   Factory for named LCG presets. All methods take an optional ``seed``
   parameter and return an ``LCGGenerator`` instance.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Method
        - Parameters
      * - ``numerical_recipes(seed)``
        - a=1664525, c=1013904223, m=2\ :sup:`32`
      * - ``park_miller(seed)``
        - a=16807, c=0, m=2\ :sup:`31`-1  (Lehmer)
      * - ``minstd_rand(seed)``
        - a=48271, c=0, m=2\ :sup:`31`-1
      * - ``minstd_rand0(seed)``
        - a=16807, c=0, m=2\ :sup:`31`-1
      * - ``borland(seed)``
        - Borland C++ parameters
      * - ``glibc(seed)``
        - GNU C Library parameters
      * - ``msvc(seed)``
        - Microsoft Visual C++ parameters
      * - ``randu(seed)``
        - IBM RANDU (historical; poor quality)

LFSR family
-----------

.. autoclass:: pyhdc.generation.lfsr.FibonacciLFSRGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.lfsr.GaloisLFSRGenerator
   :show-inheritance:

.. py:class:: CommonLFSRGenerators

   Factory for named LFSR presets. Methods return ``FibonacciLFSRGenerator``
   or ``GaloisLFSRGenerator`` instances with preset tap configurations for
   widths 8, 10, 12, 14, 16, 24, 32.

DLFSR family
------------

.. autoclass:: pyhdc.generation.dlfsr.FibonacciDLFSRGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.dlfsr.GaloisDLFSRGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.dlfsr.MatrixDLFSRGenerator
   :show-inheritance:

.. py:class:: CommonDLFSRGenerators

   Factory for DLFSR presets (compact configurations).

LCA family
----------

.. autoclass:: pyhdc.generation.lca.ElementaryLCAGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.lca.TotalisticLCAGenerator
   :show-inheritance:

.. py:class:: CommonLCAGenerators

   Factory for LCA presets based on selected cellular automata rules.

PCG family
----------

.. autoclass:: pyhdc.generation.pcg.PCGGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.pcg.MultiplicativePCGGenerator
   :show-inheritance:

.. py:class:: CommonPCGGenerators

   .. list-table::
      :header-rows: 1

      * - Method
        - Description
      * - ``pcg32(seed)``
        - Standard 32-bit PCG (recommended)
      * - ``pcg32_fast(seed)``
        - Multiplicative variant; slightly faster

Xorshift family
---------------

.. autoclass:: pyhdc.generation.xorshift.Xorshift32Generator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.Xorshift64Generator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.Xorshift128Generator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.XorshiftPlusGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.XorshiftStarGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.Xoroshiro128PlusGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.Xoroshiro128StarStarGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.Xoshiro256StarStarGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.xorshift.SplitMix64Generator
   :show-inheritance:

.. py:class:: CommonXorshiftGenerators

   .. list-table::
      :header-rows: 1

      * - Method
        - Generator class
      * - ``xorshift32(seed)``
        - Xorshift32Generator
      * - ``xorshift64(seed)``
        - Xorshift64Generator
      * - ``xorshift128(seed)``
        - Xorshift128Generator

.. py:function:: pyhdc.generation.xorshift.splitmix64_seed(seed)

   Generate N independent seeds from a single master seed using the
   SplitMix64 algorithm. Useful for initialising multiple independent streams.

   :param seed: Master seed integer.
   :returns: ``SplitMix64Generator`` that can be used to seed other generators.

ShiftedCounter family
---------------------

.. autoclass:: pyhdc.generation.shifted_counter.FeistelCounterGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.shifted_counter.ARXCounterGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.shifted_counter.SPNCounterGenerator
   :show-inheritance:

.. autoclass:: pyhdc.generation.shifted_counter.CustomMappingCounterGenerator
   :show-inheritance:

.. py:class:: CommonCounterGenerators

   Factory for ShiftedCounter presets with preset key configurations.
