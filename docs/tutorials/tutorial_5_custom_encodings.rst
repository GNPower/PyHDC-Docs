Tutorial 5: Implementing a Custom Encoding
==========================================

Every encoding in PyHDC is a thin subclass of :class:`~pyhdc.Encoding` that
returns one :class:`EncodingSpec`. The spec wires together the component
functions for generation, similarity, bundling, binding, unbinding, thinning,
and the four unary operations (permute, inverse, normalize, negative). This
tutorial builds a complete, working encoding from existing components, then
generates, bundles, binds, unbinds, compares, permutes, and inverts vectors of
your own encoding. The last sections go one level deeper: writing a component
function from scratch and wiring it into a new encoding.

**Prerequisites**: :doc:`tutorial_1_text_classification`

----

When to write a custom encoding
--------------------------------

The built-in encodings cover the standard HDC families. You write your own
subclass when you want to:

* **Swap one component** in an existing scheme. For example, pair bipolar
  multiply-binding with a different bundling rule or similarity metric.
* **Reuse the operation surface** without reimplementing it. The base class
  already handles batching, backends, broadcasting, and the dimension-first
  contract. You supply the per-element behavior and the base class does the
  rest.
* **Prototype a new encoding** by composing components before committing to a
  full implementation.

Only one method needs an override: ``_get_encoding_spec``. It is the single
abstract method on :class:`~pyhdc.Encoding`. The constructor (dimension,
backend, device, dtype, mask, generator, similarity_remap) is inherited, so a
custom encoding gets the same call surface as the built-ins.

----

The EncodingSpec fields
-----------------------

:class:`EncodingSpec` is a dataclass with seven required fields and six fields
that carry defaults. The required fields wire the core operations. The
defaulted fields cover the bit mask, the generator output contract, and the
four unary operations.

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Field
     - Default
     - Meaning
   * - ``dtype``
     - required
     - Element data type (e.g. ``np.int32``, ``np.float32``).
   * - ``element_generator``
     - required
     - Callable that draws random element values for one vector.
   * - ``similarity_fn``
     - required
     - Similarity metric, reduced over axis 0.
   * - ``bundling_fn``
     - required
     - Bundling (superposition) rule.
   * - ``thinning_fn``
     - required
     - Thinning rule, or ``NoThin`` when the family does not thin.
   * - ``binding_fn``
     - required
     - Binding rule.
   * - ``unbinding_fn``
     - required
     - Unbinding rule (set to ``RaiseNotImplementedError`` to forbid it).
   * - ``mask``
     - ``None``
     - Integer bit mask; used by ``MAP_I_Bits``, ignored elsewhere.
   * - ``generator_output_type``
     - ``"floats"``
     - ``"floats"``, ``"bits"``, or ``"words"``. The output a custom
       generator must supply.
   * - ``permute_fn``
     - ``None``
     - Permutation. ``None`` falls back to the shared ``CyclicShift``.
   * - ``inverse_fn``
     - raises
     - Binding inverse. Left unset, ``inverse()`` raises ``NotImplementedError``.
   * - ``normalize_fn``
     - raises
     - Convert to entry distribution. Left unset, ``normalize()`` raises ``NotImplementedError``.
   * - ``negative_fn``
     - raises
     - Additive (bundling) inverse. Left unset, ``negative()`` raises
       ``NotImplementedError``.

The defaulted unary fields are the part most people miss. ``inverse_fn``,
``normalize_fn``, and ``negative_fn`` default to a function that raises
``NotImplementedError``. If you want ``inverse()``, ``normalize()``, or
``negative()`` to work on your encoding, you must wire them. Leaving a field
unset is how the built-in families mark an operation as unsupported. ``MAP_C``,
for instance, sets no ``inverse_fn``, so calling ``inverse()`` on a MAP_C vector
raises an exception. ``permute_fn`` is different, ``None`` is a working default, because the
shared ``CyclicShift`` is encoding-agnostic and every built-in uses it.

----

Building the encoding
---------------------

The example below is a minimal bipolar Multiply-Add-Permute scheme, assembled
from the same components the built-in ``MAP_I`` uses. Elements are drawn from
``{-1, +1}``, binding is element-wise multiplication (which is its own inverse
for bipolar values), bundling is element-wise addition, and similarity is
cosine. The four unary fields are wired so that permute, inverse, normalize,
and negative all work.

.. code-block:: python

   import numpy as np

   from pyhdc.encodings.base import Encoding
   from pyhdc.hypervector import EncodingSpec
   from pyhdc.components.elements import BernoulliBipolar
   from pyhdc.components.binding import ElementMultiplication
   from pyhdc.components.bundling import ElementAddition
   from pyhdc.components.similarity import CosineSimilarity
   from pyhdc.components.thinning import NoThin
   from pyhdc.components.unary import (
       CyclicShift,
       IdentityInverse,
       Negate,
       SignNormalize,
   )


   class MyMAP(Encoding):
       """A minimal bipolar Multiply-Add-Permute encoding.

       Elements are drawn from {-1, +1}. Binding is element-wise
       multiplication (its own inverse), bundling is element-wise addition,
       and similarity is cosine. The 2.1.0 unary fields wire permute,
       inverse, normalize, and negative.
       """

       def _get_encoding_spec(self) -> EncodingSpec:
           return EncodingSpec(
               dtype=np.int32,
               element_generator=BernoulliBipolar,
               similarity_fn=CosineSimilarity,
               bundling_fn=ElementAddition,
               thinning_fn=NoThin,
               binding_fn=ElementMultiplication,
               unbinding_fn=ElementMultiplication,
               generator_output_type="bits",
               permute_fn=CyclicShift,
               inverse_fn=IdentityInverse,
               normalize_fn=SignNormalize,
               negative_fn=Negate,
           )

A few notes on the choices:

* ``element_generator=BernoulliBipolar`` draws each element from ``{-1, +1}``
  with equal probability, so ``generator_output_type="bits"`` describes what a
  custom generator would have to supply.
* ``binding_fn`` and ``unbinding_fn`` are both ``ElementMultiplication``.
  Element-wise multiply by ``{-1, +1}`` is its own inverse, so unbinding is the
  same operation as binding.
* ``inverse_fn=IdentityInverse`` matches that self-inverse property: the
  binding inverse of a bipolar vector is the vector itself.
* ``normalize_fn=SignNormalize`` sends a bundled vector (which holds integer
  sums) back to bipolar ``{-1, 0, +1}`` by taking the sign.
* ``negative_fn=Negate`` is element-wise negation, the additive inverse used by
  bundling.
* ``permute_fn=CyclicShift`` is set here to show the field; passing ``None``
  would select the same shared ``CyclicShift`` automatically.

----

Generating and inspecting vectors
----------------------------------

Construct the encoding like any built-in and generate vectors. Single vectors
are ``(D,)``, a batch is dimension-first, so ``size=(D, N)`` returns ``(D, N)``
with each column one hypervector.

.. code-block:: python

   enc = MyMAP(dimension=10_000)

   a = enc.generate()
   b = enc.generate()
   print("single shape:", a.data.shape)         # (10000,)

   batch = enc.generate(size=(10_000, 5))
   print("batch shape: ", batch.data.shape)     # (10000, 5)

   # Each element is bipolar.
   print(set(np.unique(a.data)))                # {-1, 1}

----

Bundle, bind, and unbind
------------------------

Bundling superposes vectors, the result stays similar to every input. Binding
combines two vectors into one that is dissimilar to both, and unbinding
recovers a component. Because element-wise multiply is exactly self-inverse for
bipolar values, ``unbind`` returns the partner without approximation.

.. code-block:: python

   # Bundle: the superposition is similar to both inputs.
   ab = enc.bundle(a, b)
   print("sim(a, a+b):", round(float(a.similarity(ab)), 4))   # ~= 0.63
   print("sim(b, a+b):", round(float(b.similarity(ab)), 4))   # ~= 0.63

   # Bind then unbind recovers the partner exactly.
   bound     = a.bind(b)
   recovered = bound.unbind(b)
   print("exact recovery:", np.array_equal(recovered.data, a.data))   # True

   # Unrelated vectors are near-orthogonal under cosine.
   print("sim(a, b):", round(float(a.similarity(b)), 4))      # ~= 0.0

The bundle-similarity scores hover near 0.63 because each of the two inputs
contributes half the superposition. They are not fixed, since ``ElementAddition``
randomizes coordinates whose summed value is an exact tie. The recovery check is
exact, and unrelated vectors sit near zero cosine, as expected for random
bipolar vectors of dimension 10,000.

----

Permute and inverse
-------------------

``permute`` is a cyclic shift along axis 0, a negative shift undoes a positive
one. ``inverse`` returns the binding inverse, which for this self-inverse scheme
is the vector itself. ``normalize`` and ``negative`` round out the unary set.

.. code-block:: python

   # permute(k) shifts along axis 0; permute(-k) restores.
   shifted  = a.permute(3)
   restored = shifted.permute(-3)
   print("shift changed data:", not np.array_equal(shifted.data, a.data))  # True
   print("inverse shift restored:", np.array_equal(restored.data, a.data)) # True

   # inverse() of a self-inverse binding returns the vector unchanged.
   print("inverse is identity:", np.array_equal(a.inverse().data, a.data)) # True

   # normalize() sends a bundle back to bipolar {-1, 0, +1}.
   norm = ab.normalize()
   print("normalized values:", set(np.unique(norm.data)))   # subset of {-1, 0, 1}

   # negative() is the element-wise additive inverse.
   print("negate:", np.array_equal(a.negative().data, -a.data))   # True

----

Operators
---------

The dunder operators dispatch straight through the encoding, so they raise or
succeed per the components you wired. For this encoding ``*``, ``/``, ``~``,
``>>``, and ``<<`` are all deterministic and match their method forms:

.. code-block:: python

   assert np.array_equal((a * b).data, a.bind(b).data)        # bind
   assert np.array_equal((bound / b).data, bound.unbind(b).data)  # unbind
   assert np.array_equal((~a).data, a.inverse().data)         # inverse
   assert np.array_equal((a >> 3).data, a.permute(3).data)    # permute +3
   assert np.array_equal((a << 3).data, a.permute(-3).data)   # permute -3

   # a + b also routes to bundle, but ElementAddition randomizes tie
   # coordinates, so a fresh draw differs run to run while staying similar
   # to both inputs.
   plus = a + b
   assert a.similarity(plus) > 0.5 and b.similarity(plus) > 0.5

The bundling operator ``+`` is the one to watch. It routes to ``bundle``, and
``ElementAddition`` redraws coordinates that sum to an exact tie, so ``a + b``
and ``a.bundle(b)`` produce different (but equally valid) vectors on separate
calls. The bind, unbind, inverse, and permute paths have no such randomness, so
their operator and method forms are byte-for-byte identical.

----

Forbidding an operation
-----------------------

To mark an operation as unsupported, leave its field unset. The default for
``inverse_fn``, ``normalize_fn``, and ``negative_fn`` is a function that raises
``NotImplementedError`` with a clear message. For example, dropping
``inverse_fn`` from the spec above makes ``inverse()`` raise an exception:

.. code-block:: python

   # With inverse_fn removed from the EncodingSpec:
   try:
       a.inverse()
   except NotImplementedError as e:
       print(e)   # This operation is not implemented for this encoding scheme.

This is exactly how the built-ins draw their support lines. ``MAP_C`` omits
``inverse_fn``, ``FHRR`` omits ``negative_fn``, ``BSC`` omits both
``normalize_fn`` and ``negative_fn``, and the four BSDC variants omit all three.
See :doc:`../user_manual/encodings_overview` for the full per-family support
table.

----

Writing a custom component function
-----------------------------------

So far you have composed *existing* components. When no built-in function does what you
need, write the component function yourself and wire it into the spec the same
way. A component is a plain function, not a class, and the spec just holds a
reference to it.

This section writes a custom bundling function and uses it to build ``MAP_S``:
``MAP_C`` with addition bundling swapped for subtraction. Subtraction is not a
meaningful superposition (the result tracks the first input and rejects the
rest), so ``MAP_S`` is not an encoding you would actually use. It is, however, the smallest
change that forces you to write a real component, which is the point.

**The bundling contract.** A bundling function takes the operands as ``*args``,
accepts a keyword-only ``axis``, and returns the folded array. Its first line
calls ``_normalize_bundling``, which turns the mixed inputs (loose ``(D,)``
vectors, a ``(D, N)`` batch, or a ``(D, N, M)`` tensor) into one dimension-first
``batch`` plus the ``reduce_axes`` to fold. Axis 0 is always the dimension ``D``
and is never reduced.

.. code-block:: python

   import numpy as np
   from pyhdc.components.input_formatting import _normalize_bundling

   try:
       import torch
   except ImportError:
       torch = None


   def ElementSubtraction(*hypervectors, axis=None):
       """Toy bundling: the first vector minus the sum of the rest, clipped to
       [-1, 1]. This is MAP_C's addition bundling with the sum swapped for
       subtraction. It has no HDC meaning and is an example only.
       """
       batch, is_torch, _, reduce_axes = _normalize_bundling(
           *hypervectors, axis=axis
       )
       if len(reduce_axes) != 1:
           raise ValueError("ElementSubtraction reduces a single batch axis")
       ax = reduce_axes[0]
       n = batch.shape[ax]

       if is_torch:
           first = batch.select(ax, 0)
           rest = batch.index_select(
               ax, torch.arange(1, n, device=batch.device)
           ).sum(dim=ax)
           return torch.clamp(first - rest, -1.0, 1.0).to(batch.dtype)
       else:      
           first = np.take(batch, 0, axis=ax)
           rest = np.take(batch, np.arange(1, n), axis=ax).sum(axis=ax)
           return np.clip(first - rest, -1.0, 1.0).astype(batch.dtype)

Four things make this a correct component, and they are the same four for every
operation family:

* **Signature.** A bundling function takes ``*hypervectors`` and a keyword-only
  ``axis=None``. The base class calls ``bundling_fn(*arrays, axis=axis)``.
* **Normalize first.** ``_normalize_bundling`` returns
  ``(batch, is_torch, reference_hv, reduce_axes)``. Do not index the raw inputs
  yourself, the normalizer is what lets one function accept loose vectors, a
  batch, or a higher-rank tensor without special-casing each shape.
* **Reduce over ``reduce_axes``, keep axis 0.** Fold only the batch axes the
  normalizer handed you, so the output is still a hypervector of dimension ``D``.
  This toy reduces a single axis (the additive bundlers accept a tuple); the
  ``is_torch`` flag tells you which backend's operations to call.
* **Return type.** Return the folded array, shape ``(D, *survivors)``. You may
  instead return ``(array, metadata_dict)``. The base class unpacks both forms
  and attaches the dict to the result's metadata. ``ElementAddition`` uses the
  tuple form to report its tie-randomization count.

Now wire it into a spec. ``MAP_S`` is ``MAP_C`` field for field, with
``bundling_fn`` pointing at the new function:

.. code-block:: python

   from pyhdc.encodings.base import Encoding
   from pyhdc.hypervector import EncodingSpec
   from pyhdc.components.elements import UniformBipolar
   from pyhdc.components.binding import ElementMultiplication
   from pyhdc.components.similarity import CosineSimilarity
   from pyhdc.components.thinning import NoThin
   from pyhdc.components.unary import Negate, SignNormalize


   class MAP_S(Encoding):
       """MAP_C with subtraction bundling. A teaching example, not a usable
       encoding."""

       def _get_encoding_spec(self) -> EncodingSpec:
           return EncodingSpec(
               dtype=np.float32,
               element_generator=UniformBipolar,
               similarity_fn=CosineSimilarity,
               bundling_fn=ElementSubtraction,     # the one swapped field
               thinning_fn=NoThin,
               binding_fn=ElementMultiplication,
               unbinding_fn=ElementMultiplication,
               generator_output_type="floats",
               normalize_fn=SignNormalize,
               negative_fn=Negate,
           )


   enc = MAP_S(dimension=10_000)
   a, b = enc.generate(), enc.generate()

   bundled = enc.bundle(a, b)            # calls ElementSubtraction
   print("bundle shape:", bundled.data.shape)            # (10000,)
   print("is clip(a - b):",
         np.array_equal(bundled.data,
                        np.clip(a.data - b.data, -1, 1).astype(np.float32)))  # True

   batch = enc.generate(size=(10_000, 4))
   print("batch bundle:", enc.bundle(batch).data.shape)  # (10000,)

Everything except bundling comes from the MAP_C component set, so binding,
unbinding, similarity, normalize, and negative behave exactly as they do for
``MAP_C``. Only ``bundle`` runs your code. ``MAP_C`` sets no ``inverse_fn``, so
``MAP_S`` inherits that gap too and ``inverse()`` raises an exception.

----

The contract for every operation family
----------------------------------------

A custom function for any other operation follows the same shape: call the
family's normalizer, branch on ``is_torch``, transform or reduce the right axis,
and return an array (optionally with a metadata dict). The signature and the
normalizer are what change between families.

.. list-table::
   :header-rows: 1
   :widths: 16 26 32 26

   * - Family
     - Signature
     - Normalize with
     - Returns
   * - Bundling
     - ``f(*hvs, axis=None)``
     - ``_normalize_bundling`` to ``(batch, is_torch, ref, reduce_axes)``
     - ``(D, *survivors)``, reduce ``reduce_axes``, keep axis 0
   * - Binding / unbinding
     - ``f(*hvs)``
     - ``_normalize_binding`` to ``(operands, is_torch, ref)``
     - same-shaped array, broadcast or loop (see below)
   * - Similarity
     - ``f(*hvs, axis=None)``
     - ``_normalize_similarity`` to ``(a, b, is_torch, scalar)``
     - reduce axis 0, ``sims.item() if scalar else sims``
   * - Unary
     - ``f(data)`` (``permute`` is ``f(data, shift=1)``)
     - none, you receive the raw ``(D, *batch)`` array
     - transformed array of the same shape

All the normalizers live in ``pyhdc.components.input_formatting``. A few rules
that are easy to miss:

* **Binding takes no ``axis``.** Binding combines operands position by position,
  so there is no batch axis to fold. After ``_normalize_binding`` you usually call
  ``_broadcast_operands`` (also in ``input_formatting``) so a ``(D,)`` key binds
  against every column of a ``(D, N)`` batch. A binder that cannot act per
  coordinate (a convolution, a matrix transform) calls ``_require_single_vector``
  to reject batched input, the ``Encoding`` layer then loops it per column.
* **Similarity returns a Python ``float`` only when ``scalar`` is true**, which
  happens only when both inputs were a single ``(D,)`` vector. Every batched call
  returns an array, so end with ``return sims.item() if scalar else sims``.
* **Unary functions receive the raw array, not ``*args``.** They act
  dimension-first along axis 0 and broadcast over any trailing batch axes. Pick
  the backend with a tensor check (the built-ins use ``torch.is_tensor(data)``).
  ``permute`` also takes a ``shift`` while ``inverse``, ``negative``, and ``normalize``
  take only the array.
* **Return shape is preserved** for binding, the unary ops, and (minus axis 0)
  similarity. Only bundling collapses a batch axis.

Wire any of these into the matching ``EncodingSpec`` field exactly as you wired
``bundling_fn`` above. For the per-family math and which families define each
unary operation, see :doc:`../user_manual/unary_operations`.

----

What you built
--------------

You implemented a complete custom encoding by subclassing
:class:`~pyhdc.Encoding` and returning one :class:`EncodingSpec`:

* Wired the seven required fields (``dtype``, ``element_generator``,
  ``similarity_fn``, ``bundling_fn``, ``thinning_fn``, ``binding_fn``,
  ``unbinding_fn``) by composing existing components.
* Wired the four 2.1.0 unary fields (``permute_fn``, ``inverse_fn``,
  ``normalize_fn``, ``negative_fn``) to give your encoding a full operation
  surface.
* Generated single ``(D,)`` vectors and dimension-first ``(D, N)`` batches.
* Bundled, bound, and unbound vectors, recovering a component exactly under
  self-inverse multiply binding.
* Computed cosine similarity, confirming superposition stays near each input
  and random pairs stay near orthogonal.
* Ran permute, inverse, normalize, and negative, and saw operators dispatch
  through the encoding, including the tie-randomized behavior of ``+``.
* Wrote a component function from scratch (``ElementSubtraction``), wired it into
  a new ``MAP_S`` encoding, and learned the signature-and-return contract that
  every custom bundling, binding, similarity, and unary function follows.

----

What's next
-----------

* :doc:`../user_manual/encodings_overview` : full encoding family comparison and
  per-family operation support
* :doc:`../user_manual/components_overview` : the component catalog you compose
  from
* :doc:`../user_manual/unary_operations` : the four unary operations and which
  families define each
* :doc:`../how_to/choose_encoding` : picking the right built-in before rolling
  your own
* :doc:`tutorial_6_custom_generators` : custom generators and reproducible
  generation
