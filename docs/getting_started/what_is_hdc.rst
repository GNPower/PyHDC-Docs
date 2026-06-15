What is Hyperdimensional Computing?
====================================

Hyperdimensional Computing (HDC), also called Vector Symbolic Architectures
(VSA), is a computing paradigm that represents information as high-dimensional
vectors (typically 1,000 to 10,000 elements) and performs reasoning through
algebraic operations on those vectors.

The core idea
-------------

Imagine encoding the concept "red circle" as a single vector of 10,000
numbers. At first glance this seems wasteful. Why not just use a struct
or a dictionary?

The answer lies in the **geometry of high-dimensional spaces**. When you draw
two vectors at random from a 10,000-dimensional space, they are almost always
*nearly orthogonal* (i.e. their dot product is close to zero). This near-orthogonality
is the property that makes HDC work.

Near-orthogonality means that two independently generated random vectors are
essentially unrelated, which gives us a huge "vocabulary" of distinct
representations. With 10,000 binary dimensions there are :math:`2^{10000}`
possible vectors, far more than there are atoms in the observable universe.
So you never run out of fresh, distinct representations.

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)

   v1 = enc.generate()
   v2 = enc.generate()

   print(v1.similarity(v1))   # 1.0  # identical
   print(v1.similarity(v2))   # ~= 0.0: orthogonal (unrelated)


Why high dimension?
-------------------

For two random unit vectors :math:`\mathbf{u}` and :math:`\mathbf{v}` in
:math:`\mathbb{R}^D`, the cosine similarity is a random variable with:

.. math::

   \mathbb{E}[\cos\theta] = 0, \qquad
   \text{Var}[\cos\theta] = \frac{1}{D}

As :math:`D` grows, the variance shrinks. In 10,000 dimensions the standard
deviation of the cosine similarity between two random vectors is just 0.01.
This means that random vectors are reliably orthogonal.

The storage capacity of a hyperdimensional system scales with
dimension. A rough rule of thumb is that a single hypervector of dimension
:math:`D` can represent a set of up to approximately :math:`O(N \times ln(M))` 
distinct items before interference (false positives) becomes significant, 
where :math:`N` is the number of vectors being bundled together and :math:`M` 
is size of the codebook/alphabet.

.. code-block:: python

   import numpy as np
   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)

   # Generate 500 random hypervectors and measure pairwise similarity
   hvs = [enc.generate() for _ in range(500)]
   sims = [hvs[0].similarity(hvs[i]) for i in range(1, 500)]
   print(f"Mean similarity: {np.mean(sims):.4f}")   # ~= 0.0
   print(f"Std  similarity: {np.std(sims):.4f}")    # ~= 0.01


The three primitive operations
-------------------------------

Everything in HDC is built from three operations. Together they are sufficient
to represent and query arbitrarily complex structured information.

Bundling (superposition)
^^^^^^^^^^^^^^^^^^^^^^^^

**Bundling** combines multiple hypervectors into a single hypervector that is
*similar to all of its components*. It is the HDC equivalent of a set or a
"bag" of concepts.

.. code-block:: python

   colors = {'red': enc.generate(), 'green': enc.generate(), 'blue': enc.generate()}

   palette = colors['red'].bundle(colors['green'], colors['blue'])

   # palette is similar to each color
   for name, hv in colors.items():
       print(f"{name}: {palette.similarity(hv):.3f}")   # all ~= 0.5–0.7

Bundling in HDC is implemented differently per encoding family (MAP uses
element-wise addition, BSC uses majority vote, BSDC uses bitwise OR), but the
rule is always the same: the result is *similar to each input*.

Binding (association)
^^^^^^^^^^^^^^^^^^^^^

**Binding** combines two hypervectors into one that is *dissimilar to both
inputs*. It creates an association or ordered pair. Crucially, binding has an
inverse: you can **unbind** to recover one component if you have the other.

.. code-block:: python

   color = colors['red']
   shape = enc.generate()           # represents "circle"

   red_circle = color.bind(shape)   # association

   # red_circle is dissimilar to both color and shape alone
   print(red_circle.similarity(color))   # ~= 0.0
   print(red_circle.similarity(shape))   # ~= 0.0

   # But unbinding recovers the original
   recovered = red_circle.unbind(shape)
   print(recovered.similarity(color))   # ~= 1.0

Binding is what lets you build structured records: binding a role (key)
to a filler (value) is the HDC equivalent of a key-value pair.

Similarity (query)
^^^^^^^^^^^^^^^^^^

**Similarity** measures how related two hypervectors are, returning a scalar
in [-1, 1] (1 = identical, 0 = orthogonal/unrelated, -1 = opposite). It is
the primary way to *query* an HDC representation.

.. code-block:: python

   query_result = red_circle.unbind(shape)

   best_match = max(colors, key=lambda c: query_result.similarity(colors[c]))
   print(best_match)   # "red"

The combination of bundling, binding, and similarity is enough to build
classifiers, associative memories, sequence encoders, and more.


What you can build
------------------

**Classifier**
   Encode each training example as a hypervector, bundle all examples of the
   same class into a class prototype, then classify a new example by comparing
   it to each prototype. The model is built in a single pass through the data.

**Associative memory (hyperdimensional dictionary)**
   Store key-value pairs as ``bundle(bind(k1,v1), bind(k2,v2), ...)``.
   Query by unbinding with a key and finding the nearest value in a codebook.

**Sequence encoding**
   Encode a sequence ``[a, b, c]`` by binding each element to a position
   hypervector (``bind(a, pos0)``) and bundling the results. The position
   binding encodes order; the bundle encodes membership.

**Analogical reasoning**
   If ``bind(king, man)`` ~= ``queen_king_relation``, then
   ``unbind(bind(queen, woman), woman)`` ~= ``queen``.


HDC vs. neural networks
------------------------

HDC and deep learning are different tools, not competitors:

* **One-shot learning**: HDC class prototypes are built by bundling, not
  gradient descent. You can add a new class at inference time.
* **Interpretability**: operations are algebraic and inspectable, similarity
  scores have a clear meaning.
* **Hardware efficiency**: binary and sparse encodings map directly to
  efficient binary compute architectures.
* **Compositionality**: structured representations are built explicitly from
  primitives, not learned implicitly.

Neural networks excel at learning representations from raw data while HDC excels at
symbolic manipulation and when training data is scarce.


Encoding families in PyHDC
---------------------------

PyHDC implements two broad families of vector symbolic architectures:

**Dense continuous** (MAP, HRR, FHRR, VTB, MBAT)
   Vectors are real-valued (or complex-valued for FHRR). Similarity is measured
   by cosine similarity. Binding is reversible. These encodings are good general
   choices.

**Binary and sparse binary** (BSC, BSDC)
   Vectors are binary. BSC uses dense binary (Bernoulli p = 0.5) with XOR
   binding and Hamming similarity. The BSDC family uses sparse binary (p << 0.5)
   with bitwise OR bundling and Overlap similarity. These encodings are
   particularly suited to hardware-efficient applications.


Key terminology
---------------

.. glossary::

   Hypervector
      A single high-dimensional vector, typically 1,000 to 10,000 elements.
      In PyHDC, represented by the :class:`~pyhdc.Hypervector` class.

   Encoding
      A specification of how hypervectors are generated and how the three
      primitives (bundle, bind, similarity) are implemented. In PyHDC,
      represented by a subclass of :class:`~pyhdc.Encoding`
      (e.g., :class:`~pyhdc.MAP_C`, :class:`~pyhdc.BSC`).

   Bundle
      The superposition operation: combines multiple hypervectors into one
      that is similar to all inputs.

   Bind
      The association operation: combines two hypervectors into one that is
      dissimilar to both, but from which either can be recovered by unbinding
      with the other.

   Unbind
      The inverse of binding: given a bound result and one of the original
      hypervectors, recovers the other.

   Codebook / Item memory
      A dictionary mapping symbolic labels (e.g., "red", "circle") to random
      hypervectors. Used as the lookup table during similarity queries.

   Thinning
      A post-processing step applied to sparse binary hypervectors after
      bundling, to prevent density from growing toward 1.0.

   Density
      For binary hypervectors, the fraction of elements that are 1. Sparse
      encodings (BSDC) aim to keep density well below 0.5.


Next steps
----------

* :doc:`installation` : install PyHDC
* :doc:`quickstart` : write your first HDC program in five minutes
* :doc:`../tutorials/index` : end-to-end worked examples
* :doc:`../user_manual/hdc_theory` : the mathematics of HDC in depth
