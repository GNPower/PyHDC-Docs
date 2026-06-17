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
