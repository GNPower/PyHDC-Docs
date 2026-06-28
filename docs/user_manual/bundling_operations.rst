Bundling Operations
===================

Bundling combines multiple hypervectors into one that is similar to all
inputs.

All bundling operations are available in ``pyhdc.components.bundling``.

Reducing over a batch axis
--------------------------

Bundling reduces a batch of hypervectors along one axis and returns the
combined result. **Axis 0 is always the hypervector dimension** :math:`D`
and is never a reduce axis, reduction happens over the batch axes (axis 1
and higher).

The ``axis`` keyword on :func:`~pyhdc.bundle` selects which batch axis to
collapse. **The default is the last axis.** For a ``(D, N)`` batch this
reduces axis 1 and returns a single ``(D,)`` hypervector, matching the 2.0
behaviour. For a ``(D, N, M)`` tensor it reduces axis 2 and returns
``(D, N)``:

.. code-block:: python

   import pyhdc

   batch = pyhdc.MAP_I().generate(size=(10000, 8))      # (D, N)
   combined = pyhdc.bundle(batch)                       # (D,) over axis 1

   tensor = pyhdc.MAP_I().generate(size=(10000, 8, 4))  # (D, N, M)
   per_column = pyhdc.bundle(tensor)                    # (D, N) over axis 2
   over_axis1 = pyhdc.bundle(tensor, axis=1)            # (D, M)

Passing ``axis=0`` raises ``ValueError`` because axis 0 is the dimension
and cannot be reduced.

**Tuple of axes.** The additive, element-wise bundlers
(``ElementAddition`` and its variants, ``AnglesOfElementAddition``, and
``Disjunction``) accept a tuple of axes and fold them together in one
reduction. ``DisjunctionThinned`` (BSDC_THIN) reduces a single axis only,
because its thinning is per column. A tuple applies to a single batched
tensor, collapsing axes 1 and 2 of a ``(D, N, M)`` tensor yields a single
``(D,)`` hypervector:

.. code-block:: python

   tensor = pyhdc.MAP_I().generate(size=(10000, 8, 4))  # (D, N, M)
   flat = pyhdc.bundle(tensor, axis=(1, 2))             # (D,)

Bundling multiple separate operands requires ``(D,)`` or ``(D, N)``
inputs, an operand with three or more axes raises ``ValueError``.

ElementAddition
----------------

**Used by**: MAP_I (HRR_NoNorm internally)

Simple element-wise sum with no normalisation:

.. math::

   (\bigoplus_k \mathbf{v}_k)_i = \sum_k v_{k,i}

The result's magnitude grows with the number of bundled vectors. Similarity
decreases slightly with each additional vector.

ElementAdditionCut
-------------------

**Used by**: MAP_C

Element-wise sum followed by clipping each element back into the valid range:

.. math::

   (\bigoplus \mathbf{v})_i = \text{clip}\!\left(\sum_k v_{k,i},\; -1,\; 1\right)

ElementAdditionBits
--------------------

**Used by**: MAP_I_Bits

Element-wise sum in a wide (int64) accumulator, then a single saturating clip to
the configured signed bit-width range (default int32). The clip happens once after the 
full reduction, not per addition, so a running sum that would overflow mid-accumulation 
saturates at the bounds instead of wrapping.

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

**Randomized-bundling metadata.** The tie-randomizing bundlers report how
many coordinates were resolved by a coin flip through ``random_zone_count``.
The type follows the result shape: for a ``(D,)`` result it is a Python
``int`` (the number of tie coordinates in that single vector). For a
batched result it is a per-output-vector count array, one entry for each
bundled output. Because the value at each tie coordinate is drawn at
random, batched bundling that resolves ties has no fixed-seed guarantee,
use ``axis=`` for the reproducible vectorized form.

ElementAdditionBipolarThreshold
---------------------------------

**Used by**: MAP_B

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

DisjunctionThinned
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

The ``density`` parameter of ``DisjunctionThinned`` is supplied by the
``BSDC_THIN`` encoding from its own ``density`` constructor argument
(default 0.5).
