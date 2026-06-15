PyHDC
=====

**PyHDC** is a Python library for `Hyperdimensional Computing (HDC)
<https://en.wikipedia.org/wiki/Hyperdimensional_computing>`_ and Vector
Symbolic Architectures (VSA). It provides a clean, composable API for building
and querying hypervectors, which are high-dimensional representations that encode
meaning as geometry.

.. code-block:: python

   import pyhdc

   enc = pyhdc.MAP_C(dimension=10_000)
   color = enc.generate()   # a random 10,000-D hypervector
   shape = enc.generate()
   obj   = color.bind(shape)          # association: "colored shape"
   print(obj.unbind(shape).similarity(color))   # ~= 1.0

PyHDC supports **14 encoding schemes** (MAP, HRR, FHRR, BSC, BSDC, VTB,
MBAT, etc.), **7 random generator families** for reproducible experiments, and a
**dual NumPy / PyTorch backend** with GPU support.

----

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   getting_started/what_is_hdc
   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: How-To Guides

   how_to/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: User Manual

   user_manual/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API Reference

   api_reference/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Project

   contributing/index
   releases/changelog
   bugs_suggestions/report_bug
   bugs_suggestions/suggest_feature
   license_authors/license
   license_authors/authors
