How to Encode Sequences with Permutation
=========================================

Permutation cyclically shifts a hypervector along its dimension axis. A
permuted vector is close to orthogonal to the original, which makes a fixed
shift a position marker. Permute a symbol by its index, bundle the results,
and the order survives the collapse into a single hypervector.

Permute a single vector
-----------------------

There are three ways to permute, and all three return the same result.

**Instance method**:

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_I(dimension=10_000)
   a   = enc.generate()          # (10000,)

   shifted = a.permute(shift=3)   # permute by +3

**Encoding method**:

.. code-block:: python

   shifted = enc.permute(a, shift=3)

**Shift operators**

``a >> k`` permutes by ``+k`` positions; ``a << k``
permutes by ``-k``:

.. code-block:: python

   shifted = a >> 3    # permute by +3, same as a.permute(shift=3)
   back    = a << 3    # permute by -3, same as a.permute(shift=-3)

The shift count must be a Python or numpy integer. A ``bool`` or a ``float``
is rejected and Python raises ``TypeError``.

The round trip is exact
-----------------------

Cyclic shift is a bijection on the ``D`` coordinates, so shifting by ``-k``
undoes shifting by ``+k`` with no approximation. ``(a >> k) << k`` returns the
original vector for every encoding and every integer ``k``:

.. code-block:: python

   shifted  = a >> 3            # permute by +3
   restored = shifted << 3      # permute by -3, exact inverse

   print(restored.similarity(a))   # 1.0 (exact, not approximate)

Unlike unbinding, which carries noise from the finite dimension, the
permutation round trip recovers the original coordinates bit for bit.

Permutation works for every encoding
------------------------------------

All encodings share the same cyclic-shift permutation, so ``permute``, the
shift operators, and the exact round trip behave identically whether you use
``MAP_C``, ``HRR``, ``BSC``, or any of the BSDC variants:

.. code-block:: python

   for cls in (pyhdc.MAP_C, pyhdc.HRR, pyhdc.BSC, pyhdc.BSDC_S):
       e = cls(dimension=10_000)
       v = e.generate()
       print(e.similarity((v >> 5) << 5, v))   # 1.0 for each

Encode a sequence and recover its order
---------------------------------------

To encode an ordered sequence, permute each symbol by its position, then
bundle. Position 0 is the symbol unshifted, position 1 is shifted by 1, and so
on. The same three symbols in a different order produce a different bundle,
because each is shifted by a different amount:

.. code-block:: python

   import pyhdc

   enc      = pyhdc.MAP_I(dimension=10_000)
   alphabet = {ch: enc.generate() for ch in 'abcd'}

   def encode_sequence(seq):
       shifted = [alphabet[ch] >> pos for pos, ch in enumerate(seq)]
       return pyhdc.bundle(*shifted)

   abc = encode_sequence('abc')
   cba = encode_sequence('cba')

   print(abc.similarity(cba))   # low: same symbols, different order

To ask which symbol sits at a given position, shift the bundle back by that
position with ``<<`` and compare the result against the alphabet. The
left-shift undoes the position marker exactly, leaving the symbol at that slot
plus bundling noise from the other positions:

.. code-block:: python

   def symbol_at(seq_hv, position, codebook):
       unshifted = seq_hv << position
       return max(codebook, key=lambda ch: unshifted.similarity(codebook[ch]))

   for pos in range(3):
       print(pos, symbol_at(abc, pos, alphabet))
   # 0 a
   # 1 b
   # 2 c

The recovered symbol is the closest match in the codebook, not an exact copy:
bundling collapses several permuted symbols into one vector, so each query
carries noise from the other positions. Larger ``dimension`` and shorter
sequences raise the margin between the correct symbol and the rest.

Permutation as a role marker in records
---------------------------------------

The same mechanism tags structure. Permuting a filler before binding it to its
slot keeps two fields distinct even when they share a role vector, and the
left-shift recovers the filler before you unbind:

.. code-block:: python

   enc    = pyhdc.MAP_I(dimension=10_000)
   role   = enc.generate()
   first  = enc.generate()
   second = enc.generate()

   record = pyhdc.bundle(
       role.bind(first),          # first filler, unshifted
       role.bind(second >> 1),    # second filler, permuted by 1
   )

   recovered_second = record.unbind(role) << 1
   print(recovered_second.similarity(second))   # high

Where permutation fits among the operators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Form
     - Permutes by
     - Notes
   * - ``a.permute(shift=k)``
     - ``+k``
     - Hypervector method
   * - ``enc.permute(a, shift=k)``
     - ``+k``
     - Encoding method, same result
   * - ``a >> k``
     - ``+k``
     - Right-shift operator
   * - ``a << k``
     - ``-k``
     - Left-shift operator, exact inverse of ``>>``
   * - ``pyhdc.permute(a, shift=k)``
     - ``+k``
     - Module-level function

For the mathematics of cyclic shift and its role in sequence and structure
encoding, see the Permutation section of
:doc:`../user_manual/unary_operations`. For the full operator table,
including how ``>>`` and ``<<`` dispatch and which operands they accept, see
:doc:`operator_syntax`.
