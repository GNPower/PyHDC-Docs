How to Use Operator Syntax
===========================

Every core operation has an operator form. The operators are thin wrappers
around the named methods, so ``a + b`` and ``a.bundle(b)`` do the same thing.
Use whichever reads more clearly for the expression at hand.

Operator-to-method map
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 30 30

   * - Operator
     - Expression
     - Method
     - Operation
   * - ``+``
     - ``a + b``
     - ``a.bundle(b)``
     - Bundle
   * - ``*``
     - ``a * b``
     - ``a.bind(b)``
     - Bind
   * - ``/``
     - ``a / b``
     - ``a.unbind(b)``
     - Unbind
   * - ``~``
     - ``~a``
     - ``a.inverse()``
     - Binding inverse
   * - ``>>``
     - ``a >> k``
     - ``a.permute(shift=k)``
     - Permute (cyclic shift by ``+k``)
   * - ``<<``
     - ``a << k``
     - ``a.permute(shift=-k)``
     - Permute (cyclic shift by ``-k``)

Bundle, bind, unbind
--------------------

The binary arithmetic operators route to the encoding's bundle, bind, and
unbind functions:

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_C(dimension=10_000)
   a     = enc.generate()
   b     = enc.generate()

   bundled = a + b            # same as a.bundle(b)
   bound   = a * b            # same as a.bind(b)
   back    = bound / a        # same as bound.unbind(a)

   print(back.similarity(b))  # ~= 1.0

The operators dispatch through the encoding, so they obey the same batch rules
as the methods. ``+`` and ``*`` chain left to right:

.. code-block:: python

   c       = enc.generate()
   record  = a * b + a * c    # binds first, then bundles the two pairs

Inverse
-------

``~a`` returns the binding inverse, the same value as ``a.inverse()``:

.. code-block:: python

   enc = pyhdc.HRR(dimension=10_000)
   a   = enc.generate()

   inv = ~a                   # same as a.inverse()
   # binding a with its inverse recovers the identity
   print((a * inv).similarity(enc.generate()))

Permute with shifts
-------------------

``>>`` and ``<<`` cyclically shift along axis 0 (the dimension). A right shift
by ``k`` and a left shift by ``k`` are exact inverses, so ``(a >> 3) << 3``
recovers ``a``:

.. code-block:: python

   enc = pyhdc.MAP_C(dimension=10_000)
   a   = enc.generate()

   shifted = a >> 3           # same as a.permute(shift=3)
   back    = shifted << 3     # same as a.permute(shift=-3)

   print(back.similarity(a))  # ~= 1.0

Permute is defined for every encoding by default (as a cyclic shift),
so ``>>`` and ``<<`` never raise on the encoding family.

Right-hand-side rules
---------------------

The shift operators accept an integer count and reject everything else.
A Python ``int`` and NumPy integers (``int32``, ``int64``) all work. A ``bool``
is rejected even though ``True`` and ``False`` are integers in Python, and a
float or any non-integral value raises ``TypeError``:

.. code-block:: python

   a >> 3            # ok: Python int
   a >> np.int64(3)  # ok: NumPy integer
   a >> True         # TypeError: bool is not accepted
   a >> 3.0          # TypeError: float is not accepted

For ``+``, ``*``, and ``/`` the right operand must be a :class:`Hypervector`.
A non-``Hypervector`` operand raises a standard ``TypeError``, there are no
reflected operators, so ``other + hv`` with a non-``Hypervector`` left operand
fails the same way:

.. code-block:: python

   a + 5             # TypeError: unsupported operand type(s)
   5 + a             # TypeError: no reflected operator

Per-family caveats
------------------

The operators inherit every per-family restriction from the methods they call.
``/`` (unbind) and ``~`` (inverse) raise ``NotImplementedError`` for the
families that do not define those operations.

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Encoding
     - ``/`` unbind
     - ``~`` inverse
     - Notes
   * - MAP_C
     - Yes
     - **Raises**
     - No inverse function; ``~a`` raises ``NotImplementedError``
   * - MAP_I, MAP_I_Bits, MAP_B
     - Yes
     - Yes
     - Element-wise multiply is self-inverse
   * - HRR family, FHRR
     - Yes
     - Yes
     - Circular convolution inverse
   * - VTB, MBAT
     - Yes
     - **Raises**
     - No inverse function; ``~a`` raises ``NotImplementedError``
   * - BSC
     - Yes
     - Yes
     - XOR is self-inverse
   * - BSDC_S, BSDC_SEG, BSDC_THIN
     - Yes
     - **Raises**
     - No inverse function; ``~a`` raises ``NotImplementedError``
   * - BSDC_CDT
     - **Raises**
     - **Raises**
     - Both ``/`` and ``~`` raise ``NotImplementedError``

The non-element-wise binders (HRR circular convolution, shifting binders,
matrix binding, VTB, BSDC_CDT) are applied per column under ``*`` and ``/`` on
batched (``ndim > 1``) input, returning one batched result. See
:doc:`bind_unbind` for the per-family binding details.

See also
--------

For the method-level reference and the full list of dunder definitions, see the
Special methods section of :doc:`../api_reference/hypervector`.
