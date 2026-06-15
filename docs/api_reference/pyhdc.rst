pyhdc: Top-Level Module
=========================

.. module:: pyhdc

Version and availability
-------------------------

.. py:data:: __version__
   :type: str

   Current version string, e.g. ``"1.1.0"``.

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
   :param size: ``None`` for a single vector, ``int`` for a 1-D batch,
                or ``tuple`` for a multi-dimensional batch.
   :param use_generator: Override the encoding's generator setting.
                         ``True`` forces the custom generator; ``False``
                         forces NumPy's default.
   :returns: A :class:`Hypervector`.

   .. code-block:: python

      enc = pyhdc.MAP_C(dimension=10_000)
      hv  = pyhdc.generate(enc)
      batch = pyhdc.generate(enc, size=100)

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
