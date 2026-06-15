Contributing to PyHDC
=====================

Thank you for your interest in contributing! This page explains how to set up
a development environment, run the checks, and submit a pull request.

The canonical contributor guide is `CONTRIBUTING.md on GitHub
<https://github.com/GNPower/PyHDC/blob/main/CONTRIBUTING.md>`_.

Development setup
-----------------

**1. Clone and create a virtual environment**

.. code-block:: bash

   git clone https://github.com/GNPower/PyHDC.git
   cd PyHDC
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate

**2. Install dependencies**

.. code-block:: bash

   make install
   # or manually:
   pip install -r requirements.txt -r requirements_dev.txt -e .

**3. Install pre-commit hooks**

.. code-block:: bash

   pre-commit install

Hooks run autoflake, isort, black, pylint, and mypy on every commit.

Running checks
--------------

.. code-block:: bash

   make lint         # autoflake + isort + black + pylint
   make type-check   # mypy
   make security     # bandit
   make test         # pytest with coverage
   make bench        # performance benchmarks (slow; run manually)

Run the test suite directly:

.. code-block:: bash

   # Full suite with coverage report
   pytest --cov=pyhdc --cov-report=term-missing

   # Single file
   pytest tests/test_encodings.py -v

   # Benchmarks only
   pytest tests/benchmarks/ --benchmark-only --benchmark-autosave

Project layout
--------------

.. code-block:: text

   pyhdc/
     __init__.py          Public API
     hypervector.py       Hypervector class, EncodingSpec, BackendManager
     encodings/           All 15 encoding classes
     generation/          7 random number generator families
     components/          Similarity, binding, bundling, elements, thinning
     recovery/            Recovery algorithms (not yet public)
     types.py             Type aliases
     exceptions.py        Exception hierarchy
   tests/
     conftest.py          Shared fixtures
     test_encodings.py    Encoding tests
     test_components.py   Component tests
     test_generation.py   Generator tests
     test_exceptions.py   Exception tests
     benchmarks/          pytest-benchmark suites

Adding an encoding
-------------------

1. Create a class in the appropriate file:

   * ``pyhdc/encodings/map.py``: MAP family
   * ``pyhdc/encodings/holographic.py``: HRR / FHRR
   * ``pyhdc/encodings/binary.py``: BSC / BSDC
   * ``pyhdc/encodings/matrix.py``: VTB / MBAT

2. Subclass :class:`~pyhdc.Encoding` and implement ``_get_encoding_spec()``:

   .. code-block:: python

      from pyhdc.encodings.base import Encoding, EncodingSpec
      from pyhdc.components.binding   import ElementMultiplication
      from pyhdc.components.bundling  import ElementAdditionCut
      from pyhdc.components.similarity import CosineSimilarity
      from pyhdc.components.elements  import UniformBipolar
      from pyhdc.components.thinning  import NoThin
      import numpy as np

      class MY_ENCODING(Encoding):
          def _get_encoding_spec(self) -> EncodingSpec:
              return EncodingSpec(
                  dtype=np.float32,
                  element_generator=UniformBipolar,
                  similarity_fn=CosineSimilarity,
                  bundling_fn=ElementAdditionCut,
                  thinning_fn=NoThin,
                  binding_fn=ElementMultiplication,
                  unbinding_fn=ElementMultiplication,
                  generator_output_type="floats",
              )

3. Export from ``pyhdc/encodings/__init__.py`` and ``pyhdc/__init__.py``.

4. Add tests in ``tests/test_encodings.py`` following the existing parametrised
   pattern.

Adding a generator
-------------------

1. Create a new file in ``pyhdc/generation/``.
2. Subclass :class:`~pyhdc.generation.HDCGenerator` and implement:
   ``_configure_internal``, ``_next_bit`` (or raise ``NotImplementedError``),
   ``_next_word``, ``set_parameters``, ``get_parameters``, ``reset``.
3. Export from ``pyhdc/generation/__init__.py``.
4. Add tests in ``tests/test_generation.py``.

Code style
----------

* Formatter: **black** (line length 88)
* Import order: **isort**
* Linting: **pylint** (fail threshold: 7.0 / 10)
* Type checking: **mypy** (``--ignore-missing-imports``)

Pull request process
--------------------

1. Fork the repository and create a feature branch from ``main``.
2. Write or update tests for any changed behaviour.
3. Ensure ``make lint`` and ``make test`` pass.
4. Open a PR against ``main``; the CI suite must be green before merging.

Versioning
----------

PyHDC uses ``bump2version``. The version lives in ``pyproject.toml`` and is
mirrored in ``pyhdc/__init__.py``:

.. code-block:: bash

   bump2version patch   # 1.1.0 → 1.1.1
   bump2version minor   # 1.1.0 → 1.2.0
   bump2version major   # 1.1.0 → 2.0.0
   git push --follow-tags

A GitHub release triggers the PyPI publish workflow automatically via OIDC
Trusted Publishing.
