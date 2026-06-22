How to Make Experiments Reproducible
=====================================

``enc.generate()`` draws from NumPy's global random state by default, which
changes between Python sessions. Setting a global seed or passing a seeded 
``HDCGenerator`` produces identical hypervectors on every run.

Setting a global seed
----------------------
.. code-block:: python

   import pyhdc
   import random
   import numpy as np
   import torch

   random.seed(42)       # sets the global seed for Python's built-in random
   np.random.seed(42)   # sets the global seed NumPY seed
   if pyhdc.TORCH_AVAILABLE:
      torch.manual_seed(42)  # sets the global seed for PyTorch
      torch.cuda.manual_seed_all(42)  # sets the global seed for all CUDA devices

   enc = pyhdc.MAP_C(dimension=10_000)
   hv  = enc.generate()   # always the same for seed=42
   print(hv.data[:5])


Basic reproducibility with seeded generators
----------------------------------------------

Pass a seeded generator to the encoding constructor:

.. code-block:: python

   import pyhdc
   from pyhdc.generation import CommonLCGGenerators

   gen = CommonLCGGenerators.numerical_recipes(seed=42)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)

   hv = enc.generate()          # always the same for seed=42
   print(hv.data[:5])

Re-run the same generation by calling ``reset()`` before each run:

.. code-block:: python

   gen.reset()
   hv_run1 = enc.generate()

   gen.reset()
   hv_run2 = enc.generate()

   import numpy as np
   print(np.allclose(hv_run1.data, hv_run2.data))   # True

Building a reproducible codebook
----------------------------------

.. code-block:: python

   from pyhdc.generation import CommonPCGGenerators

   gen = CommonPCGGenerators.pcg32(seed=0)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)

   items = ['apple', 'banana', 'cherry']

   gen.reset()
   codebook = {name: enc.generate() for name in items}

Snapshotting and restoring state
----------------------------------

If you need to resume generation mid-experiment from a known point, snapshot
the state with ``get_state()``; the exact return type is
generator-specific:

.. code-block:: python

   gen.reset()
   _ = enc.generate()   # consume one vector
   state = gen.get_state()   # snapshot

   hv_a = enc.generate()

   # Restore and re-generate from snapshot
   gen.set_seed(gen._seed)  # or: recreate with same seed and advance manually
   # Note: get_state / restore API is generator-dependent; reset() is the
   # most portable option for full reproducibility

Bypassing the generator for a single call
------------------------------------------

Pass ``use_generator=False`` to generate one vector from NumPy's default
random state without advancing the custom generator:

.. code-block:: python

   hv_np = enc.generate(use_generator=False)   # uses NumPy, not the LCG

Reproducible batched generation
-------------------------------

A tuple ``size`` produces a dimension-first batch: ``generate(size=(D, N))``
returns a ``(D, N)`` tensor of ``N`` hypervectors, and
``generate(size=(D, N, M))`` returns a ``(D, N, M)`` tensor of ``N * M``
hypervectors. Axis 0 is always the dimension ``D``, the trailing axes are the
batch. Index column ``j`` as ``batch[:, j]``.

**Batched generation reproduces itself for a fixed seed and shape.** Calling
``generate(size=(D, N))`` twice under the same seed yields the same batch:

.. code-block:: python

   import numpy as np
   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)

   np.random.seed(42)
   first = enc.generate(size=(10_000, 8))

   np.random.seed(42)
   second = enc.generate(size=(10_000, 8))

   print(np.array_equal(first.data, second.data))   # True

**The i.i.d. fast path.** When ``use_generator`` is ``False`` and the encoding's
element generator draws each coordinate independently, ``generate`` draws the
whole ``(D, *batch)`` array in one vectorized call. The fast path qualifies for
these six generators: ``BernoulliBipolar``, ``BernoulliBinary``,
``UniformBipolar``, ``UniformAngles``, ``NormalReal``, and ``BernoulliSparse``.
Because it draws the batch as one block, the result is **not** value-identical to
``N`` separate ``generate(size=D)`` calls: a block draw and a per-vector loop
walk the random stream in different orders.

**Ordered and custom generators match the per-vector loop.** ``SparseSegmented``
(the ``BSDC_SEG`` generator) is segment-structured rather than i.i.d., any custom
``HDCGenerator``, and any call with ``use_generator=True``, also falls back to
the loop. For these, ``generate`` builds the batch one vector at a time, so a
seeded batch equals ``N`` successive single-vector draws:

.. code-block:: python

   import numpy as np
   from pyhdc.generation import CommonLCGGenerators

   gen = CommonLCGGenerators.numerical_recipes(seed=7)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)

   batch = enc.generate(size=(10_000, 8), use_generator=True)

   gen.reset()
   columns = [enc.generate(size=10_000, use_generator=True) for _ in range(8)]
   loop = np.stack([c.data for c in columns], axis=-1)

   print(np.array_equal(batch.data, loop))   # True

**Use** ``axis=`` **for reproducible bundling.** The deprecated ``batch_dim``
bundling carries no fixed-seed guarantee, because tie-randomizing bundlers draw
fresh random values at tie coordinates. The ``axis=`` form reduces in place
without that extra draw, so it is the reproducible and preferred.
See :ref:`similarity-batched` for the matching axis contract on the read side.

Choosing a generator for reproducibility
------------------------------------------

All built-in generator families accept a ``seed`` parameter.  Recommended
choices:

* **PCG** (``CommonPCGGenerators.pcg32``) : best statistical quality, fully
  reproducible
* **LCG** (``CommonLCGGenerators.numerical_recipes``) : simplest, most portable
* **Xorshift** (``CommonXorshiftGenerators.xorshift64``) : very fast for large
  batches

See :doc:`../user_manual/generators` for a full comparison.
