Five-Minute Quickstart
======================

The five steps below cover the core PyHDC workflow: generating hypervectors,
applying all three primitive operations, and building a tiny item memory, all
in under 30 lines of code.


Step 1: Pick an encoding
------------------------

An *encoding* defines how hypervectors are generated and how bundle, bind, and
similarity are implemented. PyHDC ships 15 encoding classes; ``MAP_B`` is the
recommended starting point; it uses bipolar values in {-1, 1} and supports all
operations including exact unbinding.

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_B(dimension=10_000)

The only required parameter is ``dimension``, the length of every hypervector
produced by this encoding. 10,000 is a common default; lower values (1,000)
are faster but noisier, higher values (50,000) are more accurate but use more
memory.


Step 2: Generate hypervectors
------------------------------

.. code-block:: python

   v = enc.generate()

   print(v)            # Hypervector(shape=(10000,), backend=numpy, encoding=MAP_B)
   print(v.shape)      # (10000,)
   print(v.dtype)      # int8
   print(v.backend)    # numpy

Each call to ``.generate()`` draws a fresh random hypervector.  Two
independently generated hypervectors are nearly orthogonal by design.

You can generate a *batch* of hypervectors in one call:

.. code-block:: python

   batch = enc.generate(size=100)
   print(batch.shape)   # (100, 10000)


Step 3: The three operations
-----------------------------

Similarity
^^^^^^^^^^

Returns a scalar in [-1, 1]. Use it to measure how related two hypervectors
are (0 ≈ unrelated, 1 = identical).

.. code-block:: python

   a = enc.generate()
   b = enc.generate()
   c = a   # same object

   print(a.similarity(b))   # ~= 0.0  # unrelated
   print(a.similarity(c))   # 1.0     # identical

Bundling
^^^^^^^^

Produces a hypervector that is *similar to all inputs*. Think of it as a fuzzy
set union.

.. code-block:: python

   x = enc.generate()
   y = enc.generate()
   z = enc.generate()

   bundle = x.bundle(y, z)

   print(bundle.similarity(x))   # ~= 0.6
   print(bundle.similarity(y))   # ~= 0.6
   print(bundle.similarity(z))   # ~= 0.6

Binding and unbinding
^^^^^^^^^^^^^^^^^^^^^

Binding produces a hypervector that is *dissimilar to both inputs*, but from
which either input can be recovered if you have the other (unbinding).

.. code-block:: python

   key   = enc.generate()
   value = enc.generate()

   record = key.bind(value)

   print(record.similarity(key))    # ~= 0.0: dissimilar to both
   print(record.similarity(value))  # ~= 0.0

   recovered = record.unbind(key)
   print(recovered.similarity(value))  # ~= 1.0: value recovered


Step 4: Build a tiny item memory
---------------------------------

An *item memory* (or codebook) is a dictionary mapping labels to hypervectors.
Here we encode five colours, bundle three of them into a "palette", and then
query which colours are in it.

.. code-block:: python

   colour_names = ['red', 'green', 'blue', 'yellow', 'purple']
   codebook = {name: enc.generate() for name in colour_names}

   # Bundle three colours into a palette
   palette = pyhdc.bundle(codebook['red'], codebook['green'], codebook['blue'])

   # Query: which colours are in the palette?
   for name, hv in codebook.items():
       sim = palette.similarity(hv)
       print(f"{name:8s}: {sim:.3f}")

   # Output:
   # red     :  0.573
   # green   :  0.568
   # blue    :  0.561
   # yellow  :  0.012   <- not in palette
   # purple  : -0.003   <- not in palette

Items in the bundle have noticeably higher similarity than items that were
not bundled. This is the fundamental query mechanism of HDC.


Step 5: Switch to PyTorch
--------------------------

The API is identical regardless of backend. Just pass ``backend="torch"``
when creating the encoding:

.. code-block:: python

   if pyhdc.TORCH_AVAILABLE:
       enc_torch = pyhdc.MAP_B(dimension=10_000, backend="torch")
       v = enc_torch.generate()
       print(v.backend)   # torch

       # GPU: requires CUDA
       enc_gpu = pyhdc.MAP_B(dimension=10_000, backend="torch", device="cuda")

You can also move an existing hypervector between backends:

.. code-block:: python

   v_numpy = enc.generate()
   v_torch = v_numpy.to_torch()
   v_back  = v_torch.to_numpy()


Putting it all together
------------------------

Here is the complete quickstart script as a single block:

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_B(dimension=10_000)

   # Three primitives
   a, b = enc.generate(), enc.generate()
   print(a.similarity(b))              # ~= 0.0
   print(a.bundle(b).similarity(a))    # ~= 0.6
   record = a.bind(b)
   recovered = record.unbind(b)
   print(recovered.similarity(a))      # ~= 1.0

   # Item memory
   colours  = {c: enc.generate() for c in ['red','green','blue','yellow','purple']}
   palette  = pyhdc.bundle(colours['red'], colours['green'], colours['blue'])
   rankings = sorted(colours, key=lambda c: palette.similarity(colours[c]), reverse=True)
   print(rankings[:3])   # ['red', 'green', 'blue'] (order may vary)


Continue here
-------------

* :doc:`../tutorials/index` : five end-to-end tutorials, starting with
  :doc:`../tutorials/tutorial_1_text_classification`
* :doc:`../how_to/choose_encoding` : how to pick the right encoding for your
  use case
* :doc:`../user_manual/encodings_overview` : a full comparison of all 15
  encoding families
