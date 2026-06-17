How to Switch Between Backends and Devices
===========================================

PyHDC supports two backends: **NumPy** (default, CPU) and **PyTorch**
(CPU or GPU), with an identical API for both.

Create an encoding with a specific backend
-------------------------------------------

.. code-block:: python

   import pyhdc

   # NumPy (default)
   enc_np  = pyhdc.MAP_C(dimension=10_000)

   # PyTorch CPU
   enc_cpu = pyhdc.MAP_C(dimension=10_000, backend="torch")

   # PyTorch GPU
   enc_gpu = pyhdc.MAP_C(dimension=10_000, backend="torch", device="cuda")
   enc_g1  = pyhdc.MAP_C(dimension=10_000, backend="torch", device="cuda:1")

Check availability before requesting PyTorch or CUDA:

.. code-block:: python

   import torch

   if pyhdc.TORCH_AVAILABLE and torch.cuda.is_available():
       enc = pyhdc.MAP_C(dimension=10_000, backend="torch", device="cuda")
   elif pyhdc.TORCH_AVAILABLE:
       enc = pyhdc.MAP_C(dimension=10_000, backend="torch")
   else:
       enc = pyhdc.MAP_C(dimension=10_000)

Convert an existing hypervector
---------------------------------

.. code-block:: python

   hv_np  = pyhdc.MAP_C(dimension=10_000).generate()

   hv_cpu = hv_np.to_torch()          # NumPy -> PyTorch CPU
   hv_gpu = hv_cpu.cuda()             # CPU -> GPU (cuda:0)
   hv_g1  = hv_gpu.to("cuda:1")       # GPU 0 -> GPU 1
   hv_cpu2 = hv_gpu.cpu()             # GPU -> CPU
   hv_np2  = hv_cpu2.to_numpy()       # PyTorch -> NumPy

Equivalently, ``.to(device)`` accepts any device string:

.. code-block:: python

   hv_gpu = hv_np.to_torch().to("cuda")

Check current backend and device
----------------------------------

.. code-block:: python

   print(hv.backend)   # "numpy" or "torch"
   print(hv.device)    # None (numpy) or "cpu" / "cuda:0" etc.

Access the raw underlying array:

.. code-block:: python

   arr = hv.data   # numpy.ndarray or torch.Tensor

Backend mismatch error
-----------------------

Mixing a NumPy hypervector with a PyTorch one in the same operation raises
``ValueError``:

.. code-block:: python

   hv_np    = pyhdc.MAP_C(dimension=10_000).generate()
   hv_torch = pyhdc.MAP_C(dimension=10_000, backend="torch").generate()

   hv_np.similarity(hv_torch)   # ValueError: backend mismatch

Fix by converting both to the same backend first:

.. code-block:: python

   hv_torch2 = hv_np.to_torch()
   sim = hv_torch2.similarity(hv_torch)   # OK
