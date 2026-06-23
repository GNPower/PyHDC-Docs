Unary Operations
================

Unary operations transform a single hypervector, with no second operand. PyHDC
defines four: ``permute``, ``inverse``, ``negative``, and ``normalize``. Each is a
method on :class:`~pyhdc.Encoding`, mirrored on :class:`~pyhdc.Hypervector`, and
exported as a module-level function (:func:`pyhdc.permute`, :func:`pyhdc.inverse`,
:func:`pyhdc.negative`, :func:`pyhdc.normalize`). All four operate dimension-first
(axis 0 is the dimension ``D``) and broadcast over any trailing batch axes, so a
``(D,)`` vector and a ``(D, N)`` batch are handled the same way.

``permute`` is defined for every encoding. ``inverse``, ``negative``, and
``normalize`` are family-specific: an encoding wires each one only where its
algebra defines it, and the call raises ``NotImplementedError`` otherwise. An
exception raise marks an operation the encoding does not support rather than returning a wrong
result. The component function behind each operation, per family, is in the
support table on :doc:`components_overview`.

Permutation
-----------

**Available for**: every encoding

Permutation cyclically shifts a hypervector along axis 0, the dimension axis. It
is encoding-agnostic: every encoding uses the same ``CyclicShift``, so ``permute``
is defined everywhere. A shift of one position maps coordinate :math:`i` to
coordinate :math:`(i + 1) \bmod D`:

.. math::

   (\text{permute}_k(\mathbf{a}))_i = a_{(i - k) \bmod D}

The permutation by :math:`k` positions is a bijection on the :math:`D`
coordinates, so it preserves both the set of element values and the vector's
norm. A permuted vector is close to orthogonal to the original, which is what
makes permutation useful for marking position and structure.

**Sequence and structure encoding.** Permutation gives you a position marker
that does not depend on any second operand. Permuting a symbol by its index
before bundling distinguishes ``[a, b, c]`` from ``[c, b, a]``: the same three
symbols at different positions produce different bundle results because each
is shifted by a different amount. The same mechanism tags roles in a structure
(permute the filler before binding it to its slot), so order survives the
collapse into a single hypervector.

**Right-shift and left-shift operators.** ``Hypervector`` exposes permutation
through the shift operators. ``a >> k`` permutes ``a`` by ``+k`` positions and
``a << k`` permutes by ``-k`` positions:

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_I(dimension=10_000)
   a = enc.generate()          # (10000,)

   shifted = a >> 3            # permute by +3
   restored = shifted << 3     # permute by -3, exact inverse

The shift count :math:`k` must be a Python or numpy integer, a ``bool`` or a
``float`` is rejected and Python raises ``TypeError``. The operators dispatch
to :func:`pyhdc.permute`, so ``a >> k`` is identical to
``enc.permute(a, shift=k)`` and ``a << k`` to ``enc.permute(a, shift=-k)``.

**Left-shift inverts right-shift exactly.** Because cyclic shift is a
bijection, shifting by :math:`-k` undoes shifting by :math:`+k` with no
approximation:

.. math::

   \text{permute}_{-k}(\text{permute}_{k}(\mathbf{a})) = \mathbf{a}

So ``(a >> k) << k`` returns the original vector for every encoding and every
integer :math:`k`.

Inverse
-------

``inverse`` returns the *binding* inverse of a hypervector: the operand that
unbinds it. The ``~a`` operator routes to :func:`pyhdc.inverse`. The inverse is
family-specific, because each binding rule inverts differently:

* **Self-inverse multiply and XOR** (``IdentityInverse``): the element is its own
  inverse, so ``inverse(a)`` returns ``a`` unchanged. Used by ``MAP_I``,
  ``MAP_I_Bits``, ``MAP_B``, and ``BSC``, where binding by a bipolar or binary
  vector is exactly undone by binding again.
* **Circular convolution** (``ReverseInverse``): keep coordinate 0 and reverse the
  remaining coordinates along axis 0, the exact involution inverse of convolution.
  Used by ``HRR``, ``HRR_NoNorm``, and ``HRR_ConstNorm``.
* **Angle binding** (``PhaseNegate``): negate the phase modulo :math:`2\pi`. Used
  by ``FHRR``.

``MAP_C`` does not define ``inverse``: its elements are continuous on
:math:`[-1, 1]`, so element-wise multiply is not exactly self-inverse. ``VTB``,
``MBAT``, and the four BSDC variants also leave it unset, so ``~a`` on those
raises ``NotImplementedError``.

Negative
--------

``negative`` returns the *bundling* (additive) inverse through ``Negate``,
element-wise negation. Bundling a vector with its negative cancels that vector's
contribution to a superposition, which is what makes it the additive inverse.
``negative`` is defined for the MAP family (``MAP_C``, ``MAP_I``, ``MAP_I_Bits``,
``MAP_B``), the HRR family (``HRR``, ``HRR_NoNorm``, ``HRR_ConstNorm``), ``VTB``,
and ``MBAT``. ``FHRR`` (a phase has no additive inverse), ``BSC``, and the BSDC
variants do not define it.

Normalize
---------

``normalize`` maps a hypervector back to its encoding's entry element
distribution. The rule depends on the family:

* **MAP** (``SignNormalize``): take the sign, sending integer or clipped sums back
  to bipolar ``{-1, 0, +1}``. Used by ``MAP_C``, ``MAP_I``, ``MAP_I_Bits``, and
  ``MAP_B``.
* **HRR family, VTB, MBAT** (``L2Normalize``): scale each vector to unit L2 length
  along axis 0, the form cosine similarity expects.
* **FHRR** (``WrapPhase``): wrap phases into the :math:`[-\pi, \pi)`
  range.

``BSC`` and the BSDC variants do not define ``normalize``.

See also
--------

* :doc:`../how_to/permute_sequences` : encoding ordered sequences and structures
  with permutation.
* :doc:`../how_to/operator_syntax` : the ``~``, ``>>``, and ``<<`` operators and
  the operands they accept.
* :doc:`components_overview` : the per-family component support table.
