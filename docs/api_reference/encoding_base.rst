Encoding Base Class
===================

.. currentmodule:: pyhdc

.. autoclass:: Encoding
   :members:
   :undoc-members:
   :show-inheritance:

Constructor parameters
-----------------------

All encoding classes share this constructor signature:

.. code-block:: python

   Encoding(
       dimension=10_000,
       backend=None,
       device=None,
       dtype=None,
       mask=None,
       generator=None,
       similarity_remap=None,
   )

.. list-table::
   :header-rows: 1
   :widths: 20 15 10 55

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dimension``
     - ``int``
     - ``10_000``
     - Number of elements per hypervector.
   * - ``backend``
     - ``str`` or ``None``
     - ``None``
     - ``"numpy"`` or ``"torch"``. ``None`` inherits the global default
       (see :func:`~pyhdc.prefer_torch` / :func:`~pyhdc.prefer_numpy`), which is
       ``"numpy"`` unless changed.
   * - ``device``
     - ``str`` or ``None``
     - ``None``
     - PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:1"``, …).
       Only meaningful when ``backend="torch"``.
   * - ``dtype``
     - dtype or ``None``
     - ``None``
     - Override the encoding's default data type. If ``None``, uses the
       type specified by ``EncodingSpec.dtype``.
   * - ``mask``
     - ``int`` or ``None``
     - ``None``
     - Bit mask for :class:`MAP_I_Bits`; sets the integer bit width.
       Ignored by all other encodings.
   * - ``generator``
     - :class:`~pyhdc.generation.HDCGenerator` or ``None``
     - ``None``
     - Custom random generator. If ``None``, uses a ``DefaultGenerator``
       backed by NumPy.
   * - ``similarity_remap``
     - ``callable`` or ``None``
     - ``None``
     - Function applied to every similarity result. E.g.,
       :func:`~pyhdc.components.similarity.remap_to_unit` to map [-1,1] → [0,1].

Properties
----------

.. py:property:: Encoding.dimension
   :no-index:
   :type: int

   Number of elements per hypervector.

.. py:property:: Encoding.backend
   :no-index:
   :type: str

   ``"numpy"`` or ``"torch"``.

.. py:property:: Encoding.device
   :no-index:
   :type: str or None

   PyTorch device string, or ``None`` for the NumPy backend.

Methods
-------

.. py:method:: Encoding.generate(size=None, backend=None, device=None, use_generator=None)
   :no-index:

   Generate one or more hypervectors.

   :param size: ``None`` -> single ``(D,)`` vector; ``int`` -> a single vector of
                that dimension; ``tuple`` ``(D, N)`` -> a dimension-first batch of
                ``N`` vectors (each column a hypervector).
   :param backend: Override the encoding's default backend for this call.
   :param device: Override the encoding's default device for this call.
   :param use_generator: ``True`` forces the custom generator; ``False``
                         forces NumPy's default; ``None`` uses the encoding's
                         setting.
   :returns: :class:`Hypervector`

.. py:method:: Encoding.zeros(size=None, backend=None, device=None)
   :no-index:

   Return a zero-valued hypervector or batch.

   :returns: :class:`Hypervector`

.. py:method:: Encoding.from_array(array, backend=None)
   :no-index:

   Wrap an existing NumPy array or PyTorch tensor as a :class:`Hypervector`.

   :param array: ``ndarray`` or ``Tensor`` with last dimension equal to
                 ``self.dimension``.
   :param backend: Override backend detection.
   :returns: :class:`Hypervector`
   :raises DimensionsNotMatchingError: If the array's last dimension ≠
                                       ``self.dimension``.

.. py:method:: Encoding.similarity(hvA, hvB=None)
   :no-index:

   Compute similarity. Accepts ``Hypervector`` objects, raw arrays, or lists.
   If ``hvB`` is omitted, ``hvA`` must be a ``(D, N)`` batch and column 0 is
   compared against each remaining column. See
   :ref:`batched calling conventions <similarity-batched>`.

   :returns: ``float``, ``ndarray``, ``Tensor``, or ``list`` depending on inputs.

.. py:method:: Encoding.bundle(*hypervectors, batch_dim=None)
   :no-index:

   Bundle hypervectors.

   :param hypervectors: Positional :class:`Hypervector` arguments, or a
                        list of lists for batched bundling.
   :param batch_dim: Axis along which to bundle for 3-D tensor inputs.
   :returns: :class:`Hypervector` or ``list[Hypervector]``

.. py:method:: Encoding.bind(*hypervectors, batch_dim=None)
   :no-index:

   Bind hypervectors.

   :returns: :class:`Hypervector` or ``list[Hypervector]``

.. py:method:: Encoding.unbind(*hypervectors, batch_dim=None)
   :no-index:

   Unbind to recover a component.

   :raises NotImplementedError: For encodings that do not support unbinding.
   :returns: :class:`Hypervector` or ``list[Hypervector]``

.. py:method:: Encoding.thin(hypervector)
   :no-index:

   Apply the encoding's thinning operation.

   :returns: :class:`Hypervector` or ``list[Hypervector]``

.. py:method:: Encoding.set_generator(generator)
   :no-index:

   Replace the encoding's generator.

   :param generator: A :class:`~pyhdc.generation.HDCGenerator` instance.

.. py:method:: Encoding.get_generator()
   :no-index:

   Return the current generator.

   :returns: :class:`~pyhdc.generation.HDCGenerator`

Abstract method (for subclassers)
----------------------------------

.. py:method:: Encoding._get_encoding_spec()
   :no-index:

   Return an :class:`EncodingSpec` that wires together the component functions
   for this encoding.

   :returns: :class:`EncodingSpec`

EncodingSpec dataclass
-----------------------

.. py:class:: EncodingSpec
   :no-index:

   Specification dataclass that links an encoding to its component functions.

   .. list-table::
      :header-rows: 1
      :widths: 30 70

      * - Field
        - Description
      * - ``dtype``
        - NumPy data type for elements (e.g., ``np.float32``, ``np.int8``)
      * - ``element_generator``
        - Callable producing random element values given ``(size, dtype)``
      * - ``similarity_fn``
        - Callable implementing the similarity metric
      * - ``bundling_fn``
        - Callable implementing bundling
      * - ``thinning_fn``
        - Callable implementing thinning (or ``NoThin`` if not applicable)
      * - ``binding_fn``
        - Callable implementing binding
      * - ``unbinding_fn``
        - Callable implementing unbinding
      * - ``mask``
        - Optional integer bit mask (used by ``MAP_I_Bits``)
      * - ``generator_output_type``
        - ``"bits"``, ``"words"``, or ``"floats"``: the output type this
          encoding requires from a custom generator

BackendManager
--------------

.. py:class:: BackendManager
   :no-index:

   Static utility for backend detection and conversion.

   .. py:staticmethod:: get_backend(array)
      :no-index:

      Return ``"numpy"`` or ``"torch"`` for the given array.

   .. py:staticmethod:: to_numpy(array)
      :no-index:

      Convert to ``numpy.ndarray``. Detaches from autograd if needed.

   .. py:staticmethod:: to_torch(array, device=None)
      :no-index:

      Convert to ``torch.Tensor`` on the specified device.

   .. py:staticmethod:: get_device(array)
      :no-index:

      Return the device string of a tensor, or ``None`` for NumPy arrays.
