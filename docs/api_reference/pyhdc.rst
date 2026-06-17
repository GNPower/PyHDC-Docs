pyhdc: Top-Level Module
=========================

.. module:: pyhdc

Version and availability
-------------------------

.. py:data:: __version__
   :type: str

   Current version string, e.g. ``"2.0.0"``.

.. py:data:: __author__
   :type: str

   Primary author identifier, ``"GNPower"``.

.. py:data:: TORCH_AVAILABLE
   :type: bool

   ``True`` if ``import torch`` succeeded at PyHDC import time.
   ``False`` if PyTorch is not installed.

   .. code-block:: python

      import pyhdc
      if pyhdc.TORCH_AVAILABLE:
          enc = pyhdc.MAP_C(backend="torch", device="cuda")

Convenience functions
----------------------

These functions are thin wrappers that delegate to the first hypervector's
encoding. They are provided for concise one-off calls.

.. py:function:: generate(encoding, size=None, use_generator=None)

   Generate one or more hypervectors using ``encoding``.

   :param encoding: An instantiated ``Encoding`` object.
   :param size: ``None`` or omitted for a single ``(D,)`` vector; an ``int`` for a
                single vector of that dimension; a ``tuple`` ``(D, N)`` for a
                dimension-first batch of ``N`` vectors (each column a hypervector).
   :param use_generator: Override the encoding's generator setting.
                         ``True`` forces the custom generator; ``False``
                         forces NumPy's default.
   :returns: A :class:`Hypervector`.

   .. code-block:: python

      enc = pyhdc.MAP_C(dimension=10_000)
      hv  = pyhdc.generate(enc)                          # (10000,)
      batch = pyhdc.generate(enc, size=(10_000, 100))    # (10000, 100)

.. py:function:: zeros(encoding, size=None)

   Return a zero-valued hypervector (or batch) for ``encoding``.

   :param encoding: An instantiated ``Encoding`` object.
   :param size: Same as for :func:`generate`.
   :returns: A :class:`Hypervector` filled with zeros.

   .. code-block:: python

      zero = pyhdc.zeros(enc)

.. py:function:: bundle(*hypervectors)

   Bundle two or more hypervectors using the encoding of the first argument.

   :param hypervectors: Two or more :class:`Hypervector` objects produced by
                        the same encoding.
   :returns: A :class:`Hypervector`.

   .. code-block:: python

      result = pyhdc.bundle(hv1, hv2, hv3)

.. py:function:: bind(*hypervectors)

   Bind two or more hypervectors using the encoding of the first argument.

   :param hypervectors: Two or more :class:`Hypervector` objects.
   :returns: A :class:`Hypervector`.

   .. code-block:: python

      result = pyhdc.bind(key, value)

.. py:function:: stack(hypervectors)

   Combine hypervectors and/or batches into one dimension-first ``(D, N)``
   batch by concatenating along the batch axis. A 1-D ``(D,)`` vector is treated
   as a single column ``(D, 1)``. Backend-agnostic (NumPy or PyTorch).

   :param hypervectors: A list of :class:`Hypervector` objects (vectors or
                        ``(D, N)`` batches) sharing a backend.
   :returns: A single :class:`Hypervector` of shape ``(D, total_columns)``.

   .. code-block:: python

      proto    = enc.generate()                    # (10000,)
      codebook = enc.generate(size=(10_000, 50))   # (10000, 50)
      combined = pyhdc.stack([proto, codebook])    # (10000, 51); proto is column 0

Global backend and device defaults
-----------------------------------

Set a process-wide default backend/device; encodings created without an
explicit ``backend`` / ``device`` argument inherit it.

.. py:function:: prefer_torch(device=None)

   Make PyTorch the default backend (optionally pinning ``device``). Raises
   ``ImportError`` if PyTorch is not installed.

.. py:function:: prefer_cuda(index=None)

   Make PyTorch on CUDA the default (``"cuda"`` or ``"cuda:{index}"``). Raises if
   PyTorch or CUDA is unavailable.

.. py:function:: prefer_numpy()

   Reset the default backend to NumPy.

.. py:function:: prefer_cpu()

   Pin the default device to CPU (relevant when the backend is PyTorch).

.. py:function:: get_default_backend()

   Return the current default backend, ``"numpy"`` or ``"torch"``.

.. py:function:: get_default_device()

   Return the current default device string, or ``None``.

   .. code-block:: python

      pyhdc.prefer_torch()                  # or pyhdc.prefer_cuda()
      enc = pyhdc.MAP_C(dimension=10_000)   # inherits the torch backend
      pyhdc.prefer_numpy()                  # reset to numpy

Encoding classes
-----------------

All encoding classes are imported at the top level:

.. hlist::
   :columns: 3

   * :class:`MAP_C`
   * :class:`MAP_I`
   * :class:`MAP_I_Bits`
   * :class:`MAP_B`
   * :class:`HRR`
   * :class:`HRR_NoNorm`
   * :class:`HRR_ConstNorm`
   * :class:`FHRR`
   * :class:`VTB`
   * :class:`MBAT`
   * :class:`BSC`
   * :class:`BSDC_CDT`
   * :class:`BSDC_S`
   * :class:`BSDC_SEG`
   * :class:`BSDC_THIN`

See :doc:`encodings` for full documentation of each class.

Exception classes
------------------

All exception classes are imported at the top level:

* :exc:`HDCException`
* :exc:`DimensionsNotMatchingError`
* :exc:`DtypesNotMatchingError`
* :exc:`GeneratorNotSupportedError`
* :exc:`RecoveryError`
* :exc:`RecoveryNotConvergedError`
* :exc:`RecoveryNotSupportedError`

See :doc:`exceptions` for details.

Generator base classes
-----------------------

* :class:`~pyhdc.generation.HDCGenerator`: abstract base for all generators
* :class:`~pyhdc.generation.DefaultGenerator`: NumPy-backed default

See :doc:`generation` for the full family listing.
