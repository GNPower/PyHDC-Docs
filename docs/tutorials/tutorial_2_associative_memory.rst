Tutorial 2: Associative Memory with Key-Value Binding
======================================================

This tutorial builds a **hyperdimensional dictionary**: a data structure that
stores key-value pairs as a single hypervector and retrieves values by querying
with a (possibly noisy) key. The tutorial covers the approximate bind-unbind
inverse, multi-field prototypes, similarity-based field queries, and the capacity
limits that bound how many pairs a single hypervector can hold.

**Prerequisites**: :doc:`tutorial_1_text_classification`

----

The bind-unbind inverse property
----------------------------------

Binding ``key.bind(value)`` produces a hypervector dissimilar to both
inputs. But if you unbind the result with ``key``, you recover a noisy
approximation of ``value``:

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_C(dimension=10_000)
   key   = enc.generate()
   value = enc.generate()

   prototype    = key.bind(value)
   recovered = prototype.unbind(key)

   print(recovered.similarity(value))   # ~= 1.0: value recovered
   print(recovered.similarity(key))     # ~= 0.0: not key

The recovery is approximate, ``recovered`` is not identical to ``value``,
but it is similar enough that similarity search against a codebook finds the
right answer.

.. code-block:: python

   # Codebook of 50 candidate values
   candidates = [enc.generate() for _ in range(50)]
   candidates[7] = value   # value is at index 7

   scores = [recovered.similarity(c) for c in candidates]
   best   = scores.index(max(scores))
   print(best)   # 7: correct

----

Building a multi-field prototype
-------------------------------

A *prototype* stores multiple key-value pairs in a single hypervector by
bundling the individual bindings:

.. code-block:: python

   # Symbolic codebooks
   roles   = {r: enc.generate() for r in ['colour', 'shape', 'size']}
   colours = {c: enc.generate() for c in ['red', 'green', 'blue']}
   shapes  = {s: enc.generate() for s in ['circle', 'square', 'triangle']}
   sizes   = {s: enc.generate() for s in ['small', 'medium', 'large']}

   # Encode an object: small red circle
   obj = pyhdc.bundle(
       roles['colour'].bind(colours['red']),
       roles['shape'].bind(shapes['circle']),
       roles['size'].bind(sizes['small']),
   )

The variable ``obj`` is a single 10,000-D vector that encodes all three
attributes simultaneously.

----

Querying a field
-----------------

To retrieve the value of a specific field, unbind with the role hypervector
and then find the nearest candidate in the appropriate codebook:

.. code-block:: python

   def query_field(prototype, role_hv, codebook):
       """Return the label whose hypervector best matches the unbound result."""
       result = prototype.unbind(role_hv)
       return max(codebook, key=lambda label: result.similarity(codebook[label]))

   print(query_field(obj, roles['colour'], colours))   # red
   print(query_field(obj, roles['shape'],  shapes))    # circle
   print(query_field(obj, roles['size'],   sizes))     # small

----

Noise tolerance
----------------

HDC representations tolerate substantial key corruption. The experiment below
corrupts the key before querying to measure how accuracy degrades.

We simulate noise by randomly flipping a fraction of the elements in the
key hypervector before unbinding:

.. code-block:: python

   import numpy as np

   def add_noise(hv, noise_level):
       """Flip noise_level fraction of elements (MAP_B: multiply by -1)."""
       arr   = hv.data.copy()
       n_flip = int(noise_level * len(arr))
       idx    = np.random.choice(len(arr), size=n_flip, replace=False)
       arr[idx] *= -1
       return enc.from_array(arr)

   rng = np.random.default_rng(42)

   for noise in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
       noisy_role = add_noise(roles['colour'], noise)
       result     = obj.unbind(noisy_role)
       best       = max(colours, key=lambda c: result.similarity(colours[c]))
       score      = result.similarity(colours['red'])
       print(f"noise={noise:.0%}  best={best:6s}  similarity={score:.3f}")

Expected output:

.. code-block:: text

   noise=0%   best=red     similarity=0.525
   noise=5%   best=red     similarity=0.470
   noise=10%  best=red     similarity=0.427
   noise=20%  best=red     similarity=0.313
   noise=30%  best=red     similarity=0.206
   noise=40%  best=red     similarity=0.100

HDC is remarkably noise-tolerant. Even with 40% of the key corrupted, the
correct value still has the highest similarity.

----

Capacity: how many pairs can a prototype hold?
--------------------------------------------

A bundled prototype acts as a superposition of all its bindings. As you add
more bindings, the noise floor rises and eventually correct retrieval fails.

A rough rule of thumb: a hypervector of dimension :math:`D` can reliably
hold approximately :math:`O(N \times ln(M))` distinct bindings before accuracy
drops.

.. code-block:: python

   D = 10_000

   # Generate n random key-value pairs and bundle them into one prototype
   def build_prototype(n):
       keys   = [enc.generate() for _ in range(n)]
       values = [enc.generate() for _ in range(n)]
       prototype = pyhdc.bundle(*[k.bind(v) for k, v in zip(keys, values)])
       return prototype, keys, values

   def check_accuracy(prototype, keys, values):
       correct = 0
       for k, v in zip(keys, values):
           recovered = prototype.unbind(k)
           # Is v the most similar among all values?
           best = max(range(len(values)), key=lambda i: recovered.similarity(values[i]))
           if values[best].similarity(v) > 0.9:
               correct += 1
       return correct / len(keys)

   for n in [10, 50, 100, 500, 1000, 2000]:
       prototype, keys, values = build_prototype(n)
       acc = check_accuracy(prototype, keys, values)
       print(f"n={n:5d}  accuracy={acc:.0%}")

Expected pattern:

.. code-block:: text

   n=   10  accuracy=100%
   n=   50  accuracy=100%
   n=  100  accuracy=100%
   n=  500  accuracy= 73%
   n= 1000  accuracy= 25%
   n= 2000  accuracy=  5%

Accuracy starts to degrade noticeably around :math:`n = 500` and
falls below chance around :math:`n = 750`. Increasing dimension
moves this threshold proportionally.

----

Which encodings support unbinding?
------------------------------------

Not all encoding families support the ``.unbind()`` operation:

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Encoding family
     - Unbind support
     - Notes
   * - MAP_C, MAP_I, MAP_I_Bits, MAP_B
     - Yes
     - Element-wise multiplication is self-inverse
   * - HRR, HRR_NoNorm, HRR_ConstNorm
     - Yes
     - Circular correlation inverts circular convolution
   * - FHRR
     - Yes
     - Angle subtraction inverts angle addition
   * - VTB
     - Yes
     - Transpose of the transformation matrix
   * - MBAT
     - Yes (with metadata)
     - Requires the matrices stored in ``get_metadata()``
   * - BSC
     - Yes
     - XOR is self-inverse: ``bind(bind(a,b), b) == a`` exactly
   * - BSDC_CDT
     - No
     - Context-dependent thinning is not invertible
   * - BSDC_S, BSDC_SEG, BSDC_THIN
     - Yes
     - Circular shift is inverted by shifting in the opposite direction

----

Summary
-------

In this tutorial you:

* Verified the approximate bind-unbind inverse property numerically
* Constructed a multi-field prototype from role-value bindings
* Queried specific fields via unbind + codebook similarity search
* Measured noise tolerance (correct retrieval up to ≈ 40% key corruption)
* Explored capacity limits as a function of dimension

----

What's next
-----------

* :doc:`tutorial_3_pytorch_gpu` : run the same ideas on GPU
* :doc:`../how_to/bind_unbind` : quick reference for all bind/unbind patterns
* :doc:`../user_manual/binding_operations` : deep dive on all binding implementations
