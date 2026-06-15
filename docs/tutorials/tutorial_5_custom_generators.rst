Tutorial 5: Custom Generators and Reproducibility
==================================================

PyHDC comes with seven random number generator families in addition to the
default NumPy-backed generator. Any of them can be plugged into an encoding
to get reproducible hypervectors, hardware-friendly generation patterns, or
specific statistical properties. The tutorial covers the ``generator=``
parameter, picking the right built-in family, generator-encoding compatibility,
and writing a custom ``HDCGenerator`` subclass.

**Prerequisites**: :doc:`tutorial_1_text_classification`

----

Why reproducibility matters
-----------------------------

In HDC research you often need to:

* **Re-run an experiment with the same codebook** : verify that a result was
  not a lucky draw from the random codebook.
* **Compare two encodings fairly** : both should use the same item memory to
  isolate the effect of the encoding change.
* **Publish reproducible baselines** : readers should be able to reproduce
  your results exactly.

Without seeded generators, ``enc.generate()`` uses NumPy's global random state,
which changes between runs unless a seed is explicitly set via ``np.random.default_rng(seed)``.

----

Using a seeded generator
--------------------------

Pass any ``HDCGenerator`` instance to the encoding's ``generator=`` parameter.
Every call to ``.generate()`` then uses that generator instead of NumPy's
default.

.. code-block:: python

   import pyhdc
   from pyhdc.generation import CommonLCGGenerators

   # Create a seeded LCG generator
   gen = CommonLCGGenerators.numerical_recipes(seed=42)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)

   hv1 = enc.generate()

   # Reset the generator and re-generate: identical result
   gen.reset()
   hv2 = enc.generate()

   print(hv1.similarity(hv2))   # 1.0: byte-for-byte identical

The key methods on any generator:

* ``reset()`` : return to the initial state (as if just constructed)
* ``get_state()`` : snapshot the current state
* ``set_seed(seed)`` : change the seed and reinitialise

----

Built-in generator families
-----------------------------

PyHDC provides seven generator families. Each comes with a ``Common*Generators``
factory object that exposes named preset configurations.

**LCG; Linear Congruential Generator**

The classic :math:`X_{n+1} = (a X_n + c) \bmod m`. Fast, simple, good for
reproducibility. Low-order bits have poor randomness, but this is rarely an
issue for HDC.

.. code-block:: python

   from pyhdc.generation import CommonLCGGenerators

   gen = CommonLCGGenerators.park_miller(seed=42)
   # Also available: numerical_recipes, minstd_rand, borland, glibc, msvc

**LFSR; Linear Feedback Shift Register**

Hardware-efficient; produces a maximal-length sequence. Generates *bits*
natively: best paired with binary or near-binary encodings (MAP_I, MAP_B, BSC, BSDC).

.. code-block:: python

   from pyhdc.generation import CommonLFSRGenerators

   gen = CommonLFSRGenerators.fibonacci_16(seed=1)

**PCG; Permuted Congruential Generator**

Better statistical quality than LCG. Highly recommended when you want
both reproducibility and good randomness.

.. code-block:: python

   from pyhdc.generation import CommonPCGGenerators

   gen = CommonPCGGenerators.pcg32(seed=42)

**Xorshift**

Very fast word-based generators. Good for continuous encodings (MAP_C, HRR)
that need float output.

.. code-block:: python

   from pyhdc.generation import CommonXorshiftGenerators

   gen = CommonXorshiftGenerators.xorshift64(seed=12345)

**LFSR, DLFSR, LCA, ShiftedCounter**

The remaining families serve more specialised use cases: see
:doc:`../user_manual/generators` for a full comparison.

----

Generator compatibility
------------------------

Each generator family produces one of three output types: **bits**, **words**,
or **floats**. Each encoding family requires a specific type.

If you pair an incompatible generator with an encoding, PyHDC raises
:exc:`~pyhdc.GeneratorNotSupportedError`.

.. code-block:: python

   from pyhdc.generation import CommonLFSRGenerators

   # LFSR produces bits: fine for MAP_B (binary encoding)
   gen     = CommonLFSRGenerators.fibonacci_16(seed=1)
   enc_ok  = pyhdc.MAP_B(dimension=10_000, generator=gen)
   enc_ok.generate()   # works

   # LFSR produces bits: MAP_C needs floats → error
   enc_bad = pyhdc.MAP_C(dimension=10_000, generator=gen)
   try:
       enc_bad.generate()
   except pyhdc.GeneratorNotSupportedError as e:
       print(f"Error: {e}")   # generator does not support floats

Compatibility table:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Generator family
     - Output type
     - Compatible encodings
   * - LCG, PCG, Xorshift, ShiftedCounter
     - words / floats
     - MAP_C, MAP_I, MAP_I_Bits, HRR, HRR_*, FHRR, VTB, MBAT, BSC, BSDC_*
   * - LFSR, DLFSR
     - bits
     - MAP_B, MAP_I, BSC, BSDC_* (binary/integer encodings)
   * - LCA
     - bits
     - MAP_B, MAP_I, BSC, BSDC_*

When in doubt, use LCG or PCG: they support all encoding families.

----

Reproducibility demo
---------------------

This snippet generates two identical codebooks by resetting the generator
between runs:

.. code-block:: python

   import pyhdc, numpy as np
   from pyhdc.generation import CommonPCGGenerators

   gen = CommonPCGGenerators.pcg32(seed=99)
   enc = pyhdc.MAP_C(dimension=1_000, generator=gen)

   items = ['apple', 'banana', 'cherry', 'date', 'elderberry']

   # First run
   gen.reset()
   codebook_1 = {name: enc.generate() for name in items}

   # Second run with same seed
   gen.reset()
   codebook_2 = {name: enc.generate() for name in items}

   # Verify identity
   for name in items:
       data_match = np.allclose(codebook_1[name].data, codebook_2[name].data)
       print(f"{name:12s}: identical = {data_match}")

Expected output:

.. code-block:: text

   apple       : identical = True
   banana      : identical = True
   cherry      : identical = True
   date        : identical = True
   elderberry  : identical = True

----

Writing a custom generator
---------------------------

Subclass :class:`~pyhdc.generation.HDCGenerator` and implement five abstract
methods. Here is a minimal example using Python's ``random`` module as the
underlying source:

.. code-block:: python

   import random
   from typing import Any, Dict, Optional
   from pyhdc.generation.base import HDCGenerator

   class PythonRandomGenerator(HDCGenerator):
       """Simple generator wrapping Python's built-in random module."""

       def __init__(self, seed: Optional[int] = None):
           super().__init__(seed)

       def _configure_internal(self) -> None:
           self._rng = random.Random(self._seed)

       def _next_bit(self) -> int:
           return self._rng.getrandbits(1)

       def _next_word(self, word_size: int = 32) -> int:
           return self._rng.getrandbits(word_size)

       def set_parameters(self, **kwargs) -> None:
           pass   # no extra parameters for this generator

       def get_parameters(self) -> Dict[str, Any]:
           return {}

       def reset(self) -> None:
           self._rng = random.Random(self._seed)

       def get_state(self) -> Any:
           return self._rng.getstate()

Use it exactly like any built-in generator:

.. code-block:: python

   gen = PythonRandomGenerator(seed=7)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)
   hv  = enc.generate()

   gen.reset()
   hv2 = enc.generate()
   print(hv.similarity(hv2))   # 1.0

When implementing ``_next_bit()`` or ``_next_word()``, raise
``NotImplementedError`` for types your generator does not support.  PyHDC
checks ``supports_bits()`` / ``supports_words()`` / ``supports_floats()``
at generate-time and raises :exc:`~pyhdc.GeneratorNotSupportedError` with a
clear message if the encoding requires an unsupported output type.

----

Summary
-------

In this tutorial you:

* Used a seeded LCG generator for reproducible codebook generation
* Explored all seven built-in generator families and their presets
* Understood the output-type compatibility requirements
* Observed ``GeneratorNotSupportedError`` when pairing incompatible generators
* Wrote a minimal custom ``HDCGenerator`` subclass

----

What's next
-----------

* :doc:`../how_to/reproducibility` : quick reference for seeded generation
* :doc:`../user_manual/generators` : in-depth generator family comparison and
  compatibility matrix
* :doc:`../api_reference/generation` : full API reference for all generator classes
