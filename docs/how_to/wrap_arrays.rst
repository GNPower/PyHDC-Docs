How to Wrap Existing Arrays as Hypervectors
=============================================

``enc.from_array()`` wraps a pre-existing NumPy array or PyTorch tensor as a
:class:`~pyhdc.Hypervector`. The typical use cases are loading saved codebooks
from disk and converting feature vectors from other libraries.

Basic usage
-----------

.. code-block:: python

   import pyhdc
   import numpy as np

   enc = pyhdc.MAP_C(dimension=10_000)

   # Wrap a NumPy array
   arr = np.random.uniform(-1, 1, size=10_000).astype(np.float32)
   hv  = enc.from_array(arr)

   print(hv.shape)    # (10000,)
   print(hv.backend)  # numpy
   print(hv.encoding) # MAP_C instance

The array must have the same last dimension as the encoding's ``dimension``:

.. code-block:: python

   bad_arr = np.zeros(5_000)
   enc.from_array(bad_arr)   # DimensionsNotMatchingError

Load a saved codebook from disk
---------------------------------

.. code-block:: python

   # Load a codebook that was saved as a NumPy .npy file
   # Shape: (dimension, num_items) -- each column is one hypervector
   data = np.load('codebook.npy')   # shape (10000, 100)

   enc      = pyhdc.MAP_C(dimension=10_000)
   codebook = enc.from_array(data)   # one (10000, 100) batch hypervector

   query    = enc.generate()
   # similarity of query against each of the 100 columns -> (100,) array
   scores   = enc.similarity(query, codebook)
   best_idx = int(scores.argmax())

Use :meth:`~pyhdc.Hypervector.select` to pick columns from the batch by index
along the batch axis, and :func:`~pyhdc.stack` to concatenate hypervectors into
one ``(D, N)`` batch:

.. code-block:: python

   subset = codebook.select([0, 2, 4])   # (10000, 3) batch
   extended = pyhdc.stack([query, codebook])   # (10000, 101), query as column 0

Wrap a higher-rank ``(D, N, M)`` tensor
----------------------------------------

The same flow extends to tensors with more than one batch axis. ``from_array``
is a thin wrapper: it auto-detects the backend and returns a
:class:`~pyhdc.Hypervector` without transposing, reshaping, or validating the
axis order. The dimension-first contract still holds: **axis 0 must equal the
encoding's** ``dimension`` (it is the hypervector dimension ``D``), and the
trailing axes are the batch.

So a ``(D, N, M)`` array holds ``N * M`` hypervectors, one per trailing-axis
column:

.. code-block:: python

   enc = pyhdc.MAP_C(dimension=10_000)

   # axis 0 is D, axes 1 and 2 are the batch -> 8 * 4 = 32 hypervectors
   data   = np.random.uniform(-1, 1, size=(10_000, 8, 4)).astype(np.float32)
   tensor = enc.from_array(data)   # one (10000, 8, 4) batch hypervector

   print(tensor.shape)   # (10000, 8, 4)

Operate on the wrapped tensor the same way you would a ``(D, N)`` batch. Index a
single column with two trailing indices, reduce along a batch axis with
``axis=``, or compare a query against every column with ``similarity``:

.. code-block:: python

   one = tensor[:, 0, 0]          # column (0, 0) -> a single (10000,) vector

   # bundle along axis 2 (the last batch axis) -> (10000, 8)
   per_row = enc.bundle(tensor, axis=2)

   # bundle along both batch axes (1, 2) -> a single (10000,) vector
   total = enc.bundle(tensor, axis=(1, 2))

   query  = enc.generate()
   # query against every column -> (8, 4) score array, one score per column
   scores = enc.similarity(query, tensor)

The trailing axes carry through every operation. Bundling with ``axis=`` reduces
the axes you name and leaves axis 0 (the dimension) intact. ``similarity``
reduces over axis 0 and returns one score per surviving trailing column.

Wrap a PyTorch tensor
----------------------

``from_array`` auto-detects whether the input is a NumPy array or PyTorch
tensor:

.. code-block:: python

   import torch

   t  = torch.randn(10_000, dtype=torch.float32)
   enc_torch = pyhdc.MAP_C(dimension=10_000, backend="torch")
   hv = enc_torch.from_array(t)

   print(hv.backend)   # torch

Extract the underlying array
-----------------------------

Access ``.data`` to get the raw NumPy array or PyTorch tensor back:

.. code-block:: python

   arr_back = hv.data   # numpy.ndarray or torch.Tensor

You can use this to pass hypervectors to libraries that do not know about
PyHDC, such as scikit-learn or matplotlib.

Dtype notes
-----------

The dtype of the wrapped array should match what the encoding expects.
Mismatches generate a warning but do not raise an error. For example,
``MAP_C`` expects ``float32``; wrapping ``float64`` will still work but may
incur an implicit conversion.
