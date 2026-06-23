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

See also
--------

Permutation, inverse, negative, and normalize are *unary* operations: they act on
a single hypervector rather than combining two. They are documented on
:doc:`unary_operations`.
