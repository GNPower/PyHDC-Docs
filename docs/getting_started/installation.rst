Installation
============

Requirements
------------

* **Python** 3.8 or newer
* **NumPy** : installed automatically as a dependency
* **PyTorch** : optional; required only for the ``"torch"`` backend and GPU support

Install from PyPI
-----------------

.. code-block:: bash

   pip install PyHDC

Verify the installation:

.. code-block:: python

   import pyhdc
   print(pyhdc.__version__)   # e.g. 1.1.0

Install with PyTorch (GPU support)
------------------------------------

PyHDC does not pull in PyTorch automatically, because the right PyTorch
variant depends on your CUDA version. First, install PyTorch for your
platform using the `official PyTorch install selector
<https://pytorch.org/get-started/locally/>`_, then install PyHDC:

.. code-block:: bash

   # Example: CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install PyHDC

Check that PyHDC can see PyTorch:

.. code-block:: python

   import pyhdc
   print(pyhdc.TORCH_AVAILABLE)   # True if PyTorch is importable

   if pyhdc.TORCH_AVAILABLE:
       enc = pyhdc.MAP_C(dimension=10_000, backend="torch", device="cuda")
       hv  = enc.generate()
       print(hv.device)   # cuda:0 (or similar)

Install from source
-------------------

For contributors or to access unreleased features:

.. code-block:: bash

   git clone https://github.com/GNPower/PyHDC.git
   cd PyHDC
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # Linux / macOS
   pip install -e ".[dev]"

See :doc:`../contributing/index` for the full contributor setup including
pre-commit hooks, linting, and testing.

Troubleshooting
---------------

**ModuleNotFoundError: No module named 'pyhdc'**
   The package was installed in a different Python environment.  Make sure you
   activated your virtual environment before installing, and that the
   ``python`` / ``pip`` commands resolve to the same interpreter.

   .. code-block:: bash

      which python   # Linux / macOS
      where python   # Windows

**"PyTorch backend requested but torch is not available"**
   PyTorch is not installed in the current environment.  Run
   ``pip install torch`` or follow the GPU install instructions above.

**Python version error**
   PyHDC requires Python 3.8 or newer.  Check your version with
   ``python --version`` and upgrade if necessary.
