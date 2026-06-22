Binding Operations
==================

Binding creates an association between two hypervectors, producing a result
that is dissimilar to both inputs but from which either can be recovered.

All binding operations are available in ``pyhdc.components.binding``.

ElementMultiplication
----------------------

**Used by**: MAP_C, MAP_I, MAP_I_Bits, MAP_B

Element-wise product of two bipolar vectors:

.. math::

   (\mathbf{a} \otimes \mathbf{b})_i = a_i \cdot b_i

For bipolar vectors (:math:`a_i, b_i \in \{-1, +1\}`), multiplication is
self-inverse: :math:`(\mathbf{a} \otimes \mathbf{b}) \otimes \mathbf{b} = \mathbf{a}`.

Properties: commutative, associative, exactly self-inverse, preserves the
magnitude of each element.

CircularConvolution and CircularCorrelation
-------------------------------------------

**Used by**: HRR, HRR_NoNorm, HRR_ConstNorm

Circular convolution in the frequency domain:

.. math::

   \mathbf{a} \circledast \mathbf{b} = \mathcal{F}^{-1}\!\left(\mathcal{F}(\mathbf{a}) \cdot \mathcal{F}(\mathbf{b})\right)

where :math:`\mathcal{F}` is the discrete Fourier transform and multiplication
is element-wise (complex product).

Unbinding uses *circular correlation*:

.. math::

   (\mathbf{a} \circledast \mathbf{b}) \star \mathbf{b} \approx \mathbf{a}

Circular correlation is the convolution with the conjugate of the second
argument, equivalent to convolving with the *reversed* vector.

Properties: commutative, approximate inverse (not exact because convolution
is not perfectly invertible in finite dimensions), preserves magnitude of
Fourier components.

ExclusiveOr (XOR)
-----------------

**Used by**: BSC

Element-wise XOR of two binary vectors:

.. math::

   (\mathbf{a} \oplus \mathbf{b})_i = a_i \oplus b_i

XOR is *exactly* self-inverse: :math:`(\mathbf{a} \oplus \mathbf{b}) \oplus \mathbf{b} = \mathbf{a}`
with no approximation error.

Properties: commutative, associative, exactly self-inverse, flips exactly 50%
of bits on average (for random inputs), preserving density.

Shifting and Inverse Shifting
------------------------------

**Used by**: BSDC_S, BSDC_THIN

Circular shift of the vector by one position:

.. math::

   (\text{shift}(\mathbf{a}))_i = a_{(i-1) \bmod D}

Unbinding inverts the shift direction:

.. math::

   (\text{unshift}(\mathbf{a}))_i = a_{(i+1) \bmod D}

Repeated binding applies successive shifts: after :math:`k` bindings, the
result is shifted by :math:`k` positions. This creates a natural positional
encoding: the same vector at different depths in a bind sequence maps to
different hypervectors.

SegmentShifting and SegmentInverseShifting
-------------------------------------------

**Used by**: BSDC_SEG

Like shifting, but the circular shift is applied independently to each of
several contiguous segments of the vector. This gives positional encoding
within each segment independently.

AdditiveContextDependentThinning (CDT)
---------------------------------------

**Used by**: BSDC_CDT

CDT binding is defined by the thinning operation applied to the sum:

.. math::

   \mathbf{a} \otimes_{\text{CDT}} \mathbf{b} = \text{thin}(\mathbf{a} + \mathbf{b})

where ``thin`` randomly zeros bits to bring density back to the initial level.

CDT is **not invertible**: there is no unbind operation because the thinning
destroys information needed to recover the original.

VectorDerivedTransformation (VTB)
----------------------------------

**Used by**: VTB

Constructs a permutation-and-sign matrix :math:`\Phi(\mathbf{a})` from the
key vector :math:`\mathbf{a}` and applies it to the value vector
:math:`\mathbf{b}`:

.. math::

   \mathbf{a} \otimes_{\text{VTB}} \mathbf{b} = \Phi(\mathbf{a})\,\mathbf{b}

Unbinding uses the transpose (which is the matrix inverse for orthogonal
matrices):

.. math::

   \Phi(\mathbf{a})^T \cdot (\mathbf{a} \otimes_{\text{VTB}} \mathbf{b}) \approx \mathbf{b}

MatrixMultiplication (MBAT)
----------------------------

**Used by**: MBAT

Generates a random matrix :math:`M \in \mathbb{R}^{D \times D}` and applies
it to the value vector:

.. math::

   \mathbf{a} \otimes_{\text{MBAT}} \mathbf{b} = M \cdot \mathbf{b}

The matrix :math:`M` is stored in the result's metadata. Unbinding uses the
matrix inverse:

.. math::

   M^{-1} \cdot (\mathbf{a} \otimes_{\text{MBAT}} \mathbf{b}) = \mathbf{b}

Because the matrix depends on the key :math:`\mathbf{a}`, the metadata must
be preserved to perform unbinding. Retrieve it with ``result.get_metadata()``.

ElementAngleAddition and ElementAngleSubtraction
-------------------------------------------------

**Used by**: FHRR

For angle-valued vectors (:math:`a_i \in [0, 2\pi)`), binding is modular
angle addition:

.. math::

   (\mathbf{a} \otimes \mathbf{b})_i = (a_i + b_i) \bmod 2\pi

Unbinding uses element-wise angle subtraction:

.. math::

   (\mathbf{a} \otimes \mathbf{b}) \oslash \mathbf{b} = (a_i + b_i - b_i) \bmod 2\pi = a_i

Angle addition is commutative, associative, and exactly invertible modulo
:math:`2\pi`.

Permutation
-----------

**Available for**: every encoding

Permutation is a unary operation that cyclically shifts a hypervector along
axis 0, the dimension axis. Unlike the family-specific binders above, it is
encoding-agnostic: all encodings use the same ``CyclicShift``, so
``permute`` is defined everywhere. A shift of one position maps coordinate
:math:`i` to coordinate :math:`(i + 1) \bmod D`:

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

Inverse and normalize as first-class operations
-----------------------------------------------

Alongside ``permute``, version 2.1.0 promotes two more unary operations to the
top level. ``inverse`` returns the binding inverse of a hypervector, the
operand that unbinds it: the ``~a`` operator routes to :func:`pyhdc.inverse`.
``normalize`` returns the canonical form for the encoding (unit L2 length for
the holographic and matrix families, wrapped phase for FHRR, bipolar signs for
MAP). Both are wired per family, for encodings that do not define them the
call raises ``NotImplementedError`` rather than returning a wrong result. The
per-family support is listed on :doc:`../api_reference/index`.
