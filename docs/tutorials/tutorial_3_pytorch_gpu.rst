Tutorial 3: GPU-Accelerated HDC with PyTorch
=============================================

PyHDC has a PyTorch backend. Every encoding can produce
``torch.Tensor``-backed hypervectors, and all operations (bundle, bind,
similarity) work identically on CPU tensors or CUDA tensors. This tutorial
covers the backend model, creating and moving GPU hypervectors, batched
generation and similarity, and how to measure the speedup.

**Prerequisites**: :doc:`tutorial_1_text_classification` (the code will be
ported to GPU in this tutorial)

----

The backend model
-----------------

A :class:`~pyhdc.Hypervector` wraps either a ``numpy.ndarray`` or a ``torch.Tensor``. 
All HDC operations dispatch to the correct backend automatically, you never call 
NumPy or PyTorch functions directly. Although, the underlying array is accessible 
via the ``.data`` property, so can be extracted at any time for use with native 
``numpy`` or ``torch`` functions.

.. code-block:: python

   import pyhdc

   # NumPy backend (default)
   enc_np  = pyhdc.MAP_B(dimension=10_000)
   hv_np   = enc_np.generate()
   print(type(hv_np.data))   # <class 'numpy.ndarray'>

   if pyhdc.TORCH_AVAILABLE:
       import torch

       # PyTorch CPU backend
       enc_cpu = pyhdc.MAP_B(dimension=10_000, backend="torch")
       hv_cpu  = enc_cpu.generate()
       print(type(hv_cpu.data))   # <class 'torch.Tensor'>
       print(hv_cpu.device)        # cpu

       # PyTorch GPU backend
       if torch.cuda.is_available():
           enc_gpu = pyhdc.MAP_B(dimension=10_000, backend="torch", device="cuda")
           hv_gpu  = enc_gpu.generate()
           print(hv_gpu.device)    # cuda:0

----

Creating a GPU encoding (with CPU fallback)
-------------------------------------------

It is good practice to fall back to CPU if CUDA is not available:

.. code-block:: python

   import pyhdc
   import torch

   def make_enc(dimension=10_000):
       if pyhdc.TORCH_AVAILABLE and torch.cuda.is_available():
           device = "cuda"
       elif pyhdc.TORCH_AVAILABLE:
           device = "cpu"
       else:
           return pyhdc.MAP_B(dimension=dimension)   # NumPy
       return pyhdc.MAP_B(dimension=dimension, backend="torch", device=device)

   enc = make_enc()
   print(enc.backend, enc.device)

----

Moving hypervectors between backends and devices
-------------------------------------------------

You can convert an existing hypervector without recreating the encoding:

.. code-block:: python

   hv_numpy = pyhdc.MAP_B(dimension=10_000).generate()

   # NumPy -> PyTorch CPU
   hv_cpu = hv_numpy.to_torch()

   # CPU -> GPU
   hv_gpu = hv_cpu.cuda()         # or .to("cuda") or .to("cuda:0")

   # GPU -> CPU
   hv_back_cpu = hv_gpu.cpu()

   # PyTorch -> NumPy
   hv_back_np = hv_back_cpu.to_numpy()

These conversions copy the data, so the original hypervector is unchanged.

----

Batched generation
-------------------

Instead of generating one hypervector at a time, generate a whole batch at
once.  On GPU this is substantially faster because the operation is fully
vectorised across the entire batch.

.. code-block:: python

   enc = make_enc()

   # Generate 10,000 hypervectors of dimension 10,000 in one call
   batch = enc.generate(size=10_000)
   print(batch.shape)    # (10000, 10000)
   print(batch.backend)  # torch
   print(batch.device)   # cuda:0  (if CUDA available)

   # Index to get a single hypervector
   hv0 = batch[0]
   print(hv0.shape)      # (10000,)

----

Batched similarity: three calling conventions
----------------------------------------------

As of v1.1.0, the ``Encoding.similarity()`` and ``Hypervector.similarity()``
methods support three calling conventions:

**1. Two 1-D hypervectors → scalar**

.. code-block:: python

   a = enc.generate()   # shape (10000,)
   b = enc.generate()   # shape (10000,)

   sim = a.similarity(b)   # float

**2. Two 2-D batches → 1-D array (per-row pairs)**

.. code-block:: python

   batch_a = enc.generate(size=100)   # shape (100, 10000)
   batch_b = enc.generate(size=100)   # shape (100, 10000)

   sims = enc.similarity(batch_a, batch_b)   # shape (100,)
   # sims[i] = similarity(batch_a[i], batch_b[i])

**3. Single 2-D batch → 1-D array (first row vs. rest)**

.. code-block:: python

   batch = enc.generate(size=101)   # shape (101, 10000)

   sims = enc.similarity(batch)     # shape (100,)
   # sims[i] = similarity(batch[0], batch[i+1])

Convention 3 is useful for nearest-neighbour search: put the query in
position 0 and the codebook in rows 1+.

.. code-block:: python

   query = enc.generate()           # shape (10000,)
   codebook = enc.generate(size=50) # shape (50, 10000)

   batch = torch.vstack([query, codebook])   # shape (51, 10000)
   sims = enc.similarity(batch)              # shape (50,)
   best_idx = sims.argmax().item()           # index of closest match in codebook

----

Porting Tutorial 1 to GPU
--------------------------

The Tutorial 1 text classifier requires only three changes to run on GPU:

.. code-block:: python

   import pyhdc, string, torch

   # Change 1: GPU encoding
   enc = pyhdc.MAP_B(dimension=10_000, backend="torch",
                     device="cuda" if torch.cuda.is_available() else "cpu")

   alphabet = string.ascii_lowercase + string.digits + '_'
   # Change 2: codebook generation, same API
   char_hv  = {ch: enc.generate() for ch in alphabet}

   def encode_word(word, enc, char_hv, n=3):
       word = word.lower().ljust(n, '_')
       trigram_hvs = []
       for i in range(len(word) - n + 1):
           t = word[i:i+n]
           hv = char_hv[t[0]].bind(char_hv[t[1]]).bind(char_hv[t[2]])
           trigram_hvs.append(hv)
       return pyhdc.bundle(*trigram_hvs)

   python_keywords = ['false', 'none', 'true', 'and', 'for', 'if', 'import',
                       'class', 'return', 'while', 'yield', 'lambda', 'def']
   english_nouns   = ['cat', 'dog', 'house', 'river', 'cloud', 'tree', 'book',
                       'chair', 'stone', 'light', 'water', 'music', 'road']

   kw_proto   = pyhdc.bundle(*[encode_word(w, enc, char_hv) for w in python_keywords])
   noun_proto = pyhdc.bundle(*[encode_word(w, enc, char_hv) for w in english_nouns])

   def classify(word):
       hv = encode_word(word, enc, char_hv)
       # Change 3: .to_numpy() before using sklearn / printing
       kw_sim   = float(hv.similarity(kw_proto))
       noun_sim = float(hv.similarity(noun_proto))
       return 'keyword' if kw_sim > noun_sim else 'noun'

   for w in ['import', 'lamp', 'yield', 'stone']:
       print(f"{w:10s} -> {classify(w)}")

The only meaningful change is ``backend="torch", device="cuda"`` on the
encoding constructor. All operations (``.bind()``, ``.bundle()``,
``.similarity()``) work identically on GPU.

----

Benchmarking
------------

GPU becomes worthwhile for large batches and high dimensions. Here is a
simple timing comparison:

.. code-block:: python

   import time, pyhdc, torch

   D = 10_000
   N = 50_000

   enc_np  = pyhdc.MAP_B(dimension=D)
   enc_gpu = pyhdc.MAP_B(dimension=D, backend="torch",
                          device="cuda" if torch.cuda.is_available() else "cpu")

   # NumPy baseline
   t0 = time.perf_counter()
   batch_np = enc_np.generate(size=N)
   t1 = time.perf_counter()
   print(f"NumPy  generate {N}x{D}: {t1-t0:.3f}s")

   # PyTorch (CPU or GPU)
   t0 = time.perf_counter()
   batch_gpu = enc_gpu.generate(size=N)
   if torch.cuda.is_available():
       torch.cuda.synchronize()
   t1 = time.perf_counter()
   print(f"Torch  generate {N}x{D}: {t1-t0:.3f}s")

   # Batched similarity
   q = enc_np.generate()

   t0 = time.perf_counter()
   _ = enc_np.similarity(q, batch_np)
   t1 = time.perf_counter()
   print(f"NumPy  similarity 1x{N}: {t1-t0:.3f}s")

   q_gpu = q.to_torch(enc_gpu.device)
   t0 = time.perf_counter()
   _ = enc_gpu.similarity(q_gpu, batch_gpu)
   if torch.cuda.is_available():
       torch.cuda.synchronize()
   t1 = time.perf_counter()
   print(f"Torch  similarity 1x{N}: {t1-t0:.3f}s")

Typical observations:

* For small batches (< 1,000 vectors), NumPy and PyTorch CPU are comparable, the
  GPU may even be *slower* due to launch overhead.
* For large batches (> 10,000 vectors), GPU similarity search is 10-100x
  faster depending on hardware.

----

Common pitfalls
----------------

**Backend mismatch**

Mixing a NumPy hypervector with a PyTorch hypervector raises ``ValueError``:

.. code-block:: python

   hv_np    = pyhdc.MAP_B(dimension=10_000).generate()
   hv_torch = pyhdc.MAP_B(dimension=10_000, backend="torch").generate()

   hv_np.similarity(hv_torch)   # ValueError: backend mismatch

Fix: convert one of them first with ``.to_torch()`` or ``.to_numpy()``.

**Extracting scalars for Python arithmetic**

Similarity on a GPU tensor returns a tensor, not a Python float.
Wrap with ``float()`` when you need a Python number:

.. code-block:: python

   sim = hv_gpu.similarity(hv_gpu2)   # torch.Tensor, shape ()
   if float(sim) > 0.8:               # convert explicitly
       ...

----

Summary
-------

In this tutorial you:

* Created GPU encodings with a CPU fallback guard
* Moved hypervectors between backends and devices
* Used all three batched similarity calling conventions
* Ported Tutorial 1 to GPU with three lines changed
* Timed the GPU speedup for large batch operations

----

What's next
-----------

* :doc:`tutorial_4_sparse_binary` : binary and sparse encodings
* :doc:`../how_to/switch_backends` : quick reference for all backend/device conversions
* :doc:`../user_manual/backends` : in-depth explanation of the dual backend architecture
