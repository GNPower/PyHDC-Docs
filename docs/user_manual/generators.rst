Random Number Generators
========================

PyHDC provides seven families of deterministic random number generators.
They plug into encodings via the ``generator=`` constructor parameter and
produce reproducible hypervectors when seeded.

The HDCGenerator interface
---------------------------

All generators implement the abstract base class
:class:`~pyhdc.generation.HDCGenerator`. Its public interface:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``generate_bits(length)``
     - Return a list of ``length`` random bits (0 or 1)
   * - ``generate_words(length, word_size=32)``
     - Return a list of ``length`` random integers of ``word_size`` bits
   * - ``generate_floats(length, min_val=-1, max_val=1)``
     - Return a list of ``length`` random floats in [min_val, max_val]
   * - ``supports_bits()``
     - True if this generator can produce bits
   * - ``supports_words()``
     - True if this generator can produce words
   * - ``supports_floats()``
     - True if this generator can produce floats
   * - ``reset()``
     - Return to the initial state as if just constructed
   * - ``get_state()``
     - Return the current internal state
   * - ``set_seed(seed)``
     - Set a new seed and reinitialise

Generator output types and encoding compatibility
--------------------------------------------------

Each generator produces one of three output types. The encoding's
``EncodingSpec.generator_output_type`` specifies which type it needs.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Generator family
     - Output type
     - Compatible encoding families
   * - LCG, PCG, Xorshift, ShiftedCounter
     - words / floats
     - All encodings (MAP_C, HRR, BSC, BSDC, FHRR, VTB, MBAT, …)
   * - LFSR, DLFSR
     - bits
     - Binary/integer encodings only: MAP_B, MAP_I, BSC, BSDC
   * - LCA
     - bits
     - Binary/integer encodings only: MAP_B, MAP_I, BSC, BSDC

LCG: Linear Congruential Generator
--------------------------------------

Formula: :math:`X_{n+1} = (a X_n + c) \bmod m`

Fast and simple, low-order bits are less
random. Sufficient for most HDC experiments.

**Classes**: ``LCGGenerator``, ``MultiplicativeLCGGenerator``

**Presets** (``CommonLCGGenerators``):

* ``numerical_recipes(seed)`` : a = 1664525, c = 1013904223, m = 2^32
* ``park_miller(seed)`` : Lehmer generator, no increment; a = 16807, m = 2^31-1
* ``minstd_rand(seed)`` : a = 48271
* ``borland(seed)`` : Borland C++ parameters
* ``glibc(seed)`` : glibc parameters
* ``msvc(seed)`` : Visual C++ parameters

LFSR: Linear Feedback Shift Register
---------------------------------------

Shifts a binary register and XORs feedback taps to generate bits.
Produces a maximal-length sequence of period :math:`2^n - 1` for an
:math:`n`-bit register. Natively produces bits so is suitable for binary and
integer encodings.

**Classes**: ``LFSRGenerator``, ``FibonacciLFSRGenerator``, ``GaloisLFSRGenerator``

**Presets** (``CommonLFSRGenerators``): preset tap configurations for widths
8, 10, 12, 14, 16, 24, 32.

DLFSR: Digit-Serial LFSR
--------------------------

A digit-serial variant of LFSR that processes multiple bits per cycle.
Useful for hardware implementations where parallelism is available.

**Classes**: ``DLFSRGenerator``, ``FibonacciDLFSRGenerator``,
``GaloisDLFSRGenerator``, ``MatrixDLFSRGenerator``

**Presets** (``CommonDLFSRGenerators``): various compact configurations.

LCA: Linear Cellular Automata
--------------------------------

A 1-D cellular automaton where each cell's next state depends on itself and
its neighbours. Simple, spatially structured, and highly parallelisable.

**Classes**: ``LCAGenerator``, ``ElementaryLCAGenerator``,
``TotalisticLCAGenerator``

**Presets** (``CommonLCAGenerators``): selected rules with good randomness
properties.

PCG: Permuted Congruential Generator
---------------------------------------

An LCG with a final permutation step that produces substantially better
statistical quality than plain LCG while keeping the same speed. Recommended
when you want both reproducibility and good randomness.

**Classes**: ``PCGGenerator``, ``MultiplicativePCGGenerator``

**Presets** (``CommonPCGGenerators``):

* ``pcg32(seed)`` : standard 32-bit PCG (recommended)
* ``pcg32_fast(seed)`` : multiplicative variant, slightly faster

Xorshift family
----------------

Very fast generators based on XOR and bit-shift operations. Excellent for
large-batch generation with continuous encodings.

**Classes** (all inherit from ``XorshiftGenerator``):
``Xorshift32Generator``, ``Xorshift64Generator``, ``Xorshift128Generator``,
``XorshiftPlusGenerator``, ``XorshiftStarGenerator``,
``Xoroshiro128PlusGenerator``, ``Xoroshiro128StarStarGenerator``,
``Xoshiro256StarStarGenerator``, ``SplitMix64Generator``

**Presets** (``CommonXorshiftGenerators``):

* ``xorshift32(seed)`` : 32-bit Xorshift
* ``xorshift64(seed)`` : 64-bit Xorshift (better period)
* ``xorshift128(seed)`` : 128-bit state (very long period)

**Utility**: ``splitmix64_seed(seed)``; generates independent seeds for
multiple generators from a single master seed.

ShiftedCounter family
-----------------------

Counter-based generators that apply a cryptographic-style scrambling function
to a monotonically increasing counter. The scrambling function (Feistel
network, ARX cipher, SPN, or custom) ensures that sequential counter values
produce seemingly random outputs.

**Classes**: ``ShiftedCounterGenerator``, ``FeistelCounterGenerator``,
``ARXCounterGenerator``, ``SPNCounterGenerator``,
``CustomMappingCounterGenerator``

**Presets** (``CommonCounterGenerators``): preset key configurations for each
variant.

When to use each family
------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Use case
     - Recommended family
   * - General-purpose reproducibility
     - PCG (``pcg32``) or LCG (``numerical_recipes``)
   * - Hardware mapping
     - LFSR or LCA (natively produce bits)
   * - High-throughput float generation
     - Xorshift (``xorshift64`` or ``xoshiro256``)
   * - Parallel generation from one seed
     - Xorshift + ``splitmix64_seed`` for independent streams
   * - Custom cryptographic properties
     - ShiftedCounter (``FeistelCounterGenerator``)
   * - Maximum simplicity and portability
     - LCG (``numerical_recipes``)
