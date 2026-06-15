Bundling Operations
===================

Bundling combines multiple hypervectors into one that is similar to all
inputs.

All bundling operations are available in ``pyhdc.components.bundling``.

ElementAddition
----------------

**Used by**: MAP_I (HRR_NoNorm internally)

Simple element-wise sum with no normalisation:

.. math::

   (\bigoplus_k \mathbf{v}_k)_i = \sum_k v_{k,i}

The result's magnitude grows with the number of bundled vectors. Similarity
decreases slightly with each additional vector because the result moves away
from the unit sphere.

ElementAdditionCut
-------------------

**Used by**: MAP_C

Element-wise sum followed by clipping each element back into the valid range:

.. math::

   (\bigoplus \mathbf{v})_i = \text{clip}\!\left(\sum_k v_{k,i},\; -1,\; 1\right)

The ``random_choice_range`` parameter controls randomised tie-breaking at the
clip boundary: useful for reducing systematic bias in the result.

ElementAdditionBits
--------------------

**Used by**: MAP_I_Bits, MAP_B

Element-wise sum with per-step clipping to the integer range determined by
the ``mask`` bit width. For MAP_B (binary), this clips to {0, 1}.

ElementAdditionBinaryThreshold
--------------------------------

**Used by**: BSC

Element-wise sum followed by a majority-vote threshold: each output element
is 1 if more than half the input elements at that position are 1, otherwise 0.

For an odd number of inputs, this is deterministic. For an even number, ties
at exactly 0.5 are resolved randomly.

.. math::

   (\bigoplus \mathbf{v})_i =
   \begin{cases}
   1 & \text{if } \sum_k v_{k,i} > n/2 \\
   0 & \text{if } \sum_k v_{k,i} < n/2 \\
   \text{rand}(\{0,1\}) & \text{if } \sum_k v_{k,i} = n/2
   \end{cases}

ElementAdditionBipolarThreshold
---------------------------------

Element-wise sum followed by a sign function, remapping to {-1, +1}.

ElementAdditionNormalized
--------------------------

**Used by**: HRR, VTB, MBAT

Element-wise sum followed by L2 normalisation:

.. math::

   \bigoplus \mathbf{v} = \frac{\sum_k \mathbf{v}_k}{\left\|\sum_k \mathbf{v}_k\right\|_2}

The result is always a unit vector, which preserves the geometric properties
needed for cosine similarity to work reliably.

ElementAdditionConstantNormalized
-----------------------------------

**Used by**: HRR_ConstNorm

Divides by :math:`\sqrt{M}` where :math:`M` is the number of bundled vectors:

.. math::

   \bigoplus_M \mathbf{v} = \frac{\sum_k \mathbf{v}_k}{\sqrt{M}}

This normalises the *expected* magnitude rather than the actual magnitude,
which gives a different noise profile than L2 normalisation.

AnglesOfElementAddition
------------------------

**Used by**: FHRR

For angle-valued vectors, bundling sums the phasors and extracts the angle
of the resultant:

.. math::

   (\bigoplus \mathbf{v})_i = \arg\!\left(\sum_k e^{j v_{k,i}}\right)

This is the circular mean of a set of angles: appropriate when values are
periodic (e.g., directions, phases).

Disjunction (bitwise OR)
-------------------------

**Used by**: BSDC_CDT, BSDC_S, BSDC_SEG

Element-wise bitwise OR of binary vectors:

.. math::

   (\bigoplus \mathbf{v})_i = \bigvee_k v_{k,i}

OR can only turn bits on, never off, so density monotonically increases with
each bundle step. After :math:`n` steps with initial density :math:`\rho`:

.. math::

   \rho_n \approx 1 - (1 - \rho)^n

After 100 bundle steps from :math:`\rho_0 = 0.01`, density reaches
:math:`1 - (0.99)^{100} \approx 0.63`. This makes all vectors indistinguishable.
Use :class:`~pyhdc.BSDC_THIN` to avoid this problem.

DisjunctionThinned (v1.1.0)
-----------------------------

**Used by**: BSDC_THIN

Bitwise OR followed by random thinning to maintain a target density:

1. Compute element-wise OR
2. Count bits that are 1 and compute actual density :math:`\rho_{\text{actual}}`
3. If :math:`\rho_{\text{actual}} > \rho_{\text{target}}`, randomly clear
   bits until density returns to :math:`\rho_{\text{target}}`

.. math::

   p(\text{clear bit } i \mid v_i = 1) = 1 - \frac{\rho_{\text{target}}}{\rho_{\text{actual}}}

This keeps density stable at the initial level regardless of how many bundle
steps are performed.

The ``target_density`` parameter of ``DisjunctionThinned`` is set
automatically by the ``BSDC_THIN`` encoding based on the initial density of
generated hypervectors.
