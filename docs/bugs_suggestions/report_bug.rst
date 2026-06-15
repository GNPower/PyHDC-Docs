Report a Bug
============

Please report bugs on the `PyHDC GitHub Issues page
<https://github.com/GNPower/PyHDC/issues>`_.

What to include
---------------

A good bug report makes it easy to reproduce and fix the problem. Please
include:

* **PyHDC version**: ``import pyhdc; print(pyhdc.__version__)``
* **Python version**: ``python --version``
* **NumPy version**: ``import numpy; print(numpy.__version__)``
* **PyTorch version** (if relevant): ``import torch; print(torch.__version__)``
* **Operating system** and CPU/GPU model (for performance or GPU bugs)
* **A minimal reproducible example**: the shortest code that reproduces
  the issue
* **Expected behaviour**: what you expected to happen
* **Actual behaviour**: what actually happened, including the full traceback

Security vulnerabilities
------------------------

Do **not** file public issues for security vulnerabilities. Please follow the
responsible disclosure process described in `SECURITY.md on GitHub
<https://github.com/GNPower/PyHDC/blob/main/SECURITY.md>`_.

Common gotchas
--------------

Before filing a bug, check whether you are hitting one of these:

**Backend mismatch**
   Mixing NumPy and PyTorch hypervectors in the same operation raises
   ``ValueError``. Convert with ``.to_torch()`` or ``.to_numpy()`` first.
   See :doc:`../how_to/switch_backends`.

**Similarity range (v1.0.x → v1.1.0 change)**
   ``HammingDistance`` and ``Overlap`` now return [-1, 1] instead of [0, 1].
   Use ``similarity_remap=remap_to_unit`` on the encoding to restore [0, 1].
   See :doc:`../how_to/similarity_remap`.

**BSDC_CDT does not support unbinding**
   ``BSDC_CDT.unbind()`` raises ``NotImplementedError`` by design; CDT
   binding is not invertible. Use :class:`~pyhdc.BSDC_S` or
   :class:`~pyhdc.BSDC_THIN` if you need unbinding.

**GeneratorNotSupportedError**
   You have paired an LFSR / LCA generator (bit output) with an encoding
   that requires float output (MAP_C, HRR, etc.). See the compatibility
   table in :doc:`../user_manual/generators`.
