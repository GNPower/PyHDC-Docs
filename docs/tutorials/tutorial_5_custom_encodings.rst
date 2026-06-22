Tutorial 5: Implementing a Custom Encoding
==========================================

Every encoding in PyHDC is a thin subclass of :class:`~pyhdc.Encoding` that
returns one :class:`EncodingSpec`. The spec wires together the component
functions for generation, similarity, bundling, binding, unbinding, thinning,
and the four unary operations (permute, inverse, normalize, negative). This
tutorial builds a complete, working encoding from existing components, then
generates, bundles, binds, unbinds, compares, permutes, and inverts vectors of
your own encoding.

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

----

What's next
-----------

* :doc:`../user_manual/encodings_overview` : full encoding family comparison and
  per-family operation support
* :doc:`../user_manual/components_overview` : the component catalog you compose
  from
* :doc:`../how_to/choose_encoding` : picking the right built-in before rolling
  your own
* :doc:`tutorial_6_custom_generators` : custom generators and reproducible
  generation
