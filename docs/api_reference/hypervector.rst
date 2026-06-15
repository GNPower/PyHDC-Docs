Hypervector
===========

.. currentmodule:: pyhdc

.. autoclass:: Hypervector
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: data, encoding, backend

Reference summary
------------------

Constructor
^^^^^^^^^^^

.. code-block:: python

   Hypervector(data, encoding, backend=None, metadata=None)

Normally you do not construct ``Hypervector`` directly: use
``encoding.generate()``, ``encoding.zeros()``, or ``encoding.from_array()``.

Properties
^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Property
     - Type
     - Description
   * - ``.data``
     - ``ndarray`` or ``Tensor``
     - Underlying NumPy array or PyTorch tensor (read-only)
   * - ``.encoding``
     - ``Encoding``
     - The encoding that produced this hypervector
   * - ``.backend``
     - ``str``
     - ``"numpy"`` or ``"torch"``
   * - ``.shape``
     - ``tuple``
     - Shape of the underlying array (e.g., ``(10000,)`` or ``(100, 10000)``)
   * - ``.ndim``
     - ``int``
     - Number of dimensions (1 for a single HV, 2 for a batch)
   * - ``.dtype``
     - dtype
     - Data type of the underlying array (e.g., ``float32``, ``int8``)
   * - ``.device``
     - ``str`` or ``None``
     - PyTorch device string (e.g., ``"cuda:0"``); ``None`` for NumPy backend

Similarity
^^^^^^^^^^

.. py:method:: Hypervector.similarity(other)
   :no-index:

   Compute similarity to another hypervector (or batch).

   Delegates to the encoding's similarity function. Returns a value in [-1, 1].

   :param other: A :class:`Hypervector` of the same encoding and backend.
   :returns: ``float``, ``ndarray``, or ``Tensor`` depending on input shapes.
             See :ref:`batched calling conventions <similarity-batched>`.

Bundle and bind
^^^^^^^^^^^^^^^

.. py:method:: Hypervector.bundle(*others, batch_dim=None)
   :no-index:

   Bundle this hypervector with one or more others.

   :param others: Additional :class:`Hypervector` objects.
   :param batch_dim: For 3-D inputs, the axis along which to bundle.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.bind(*others, batch_dim=None)
   :no-index:

   Bind this hypervector with one or more others.

   :param others: Additional :class:`Hypervector` objects.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.unbind(*others, batch_dim=None)
   :no-index:

   Unbind to recover a component.

   :raises NotImplementedError: For encodings that do not support unbinding
                                (e.g., ``BSDC_CDT``).
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.thin()
   :no-index:

   Apply the encoding's thinning operation (sparse binary encodings only).
   For encodings that do not thin, returns ``self`` unchanged.

   :returns: :class:`Hypervector`

Backend and device conversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: Hypervector.to_numpy()
   :no-index:

   Convert to NumPy backend. Copies data if currently on PyTorch.

   :returns: :class:`Hypervector`

.. py:method:: Hypervector.to_torch(device=None)
   :no-index:

   Convert to PyTorch backend.

   :param device: Target device string (``"cpu"``, ``"cuda"``, etc.).
                  ``None`` means CPU.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.to(device)
   :no-index:

   Move to a specific device. Equivalent to ``.to_torch(device)`` when on
   NumPy, or ``.data.to(device)`` when already on PyTorch.

   :param device: Device string or ``torch.device``.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.cpu()
   :no-index:

   Move to CPU. Equivalent to ``.to("cpu")``.

   :returns: :class:`Hypervector`

.. py:method:: Hypervector.cuda(device=None)
   :no-index:

   Move to CUDA. Equivalent to ``.to("cuda")`` or ``.to("cuda:<device>")``.

   :param device: Optional CUDA device index (int).
   :returns: :class:`Hypervector`

Metadata
^^^^^^^^

.. py:method:: Hypervector.get_metadata()
   :no-index:

   Return the metadata dictionary attached to this hypervector.

   Most hypervectors have an empty metadata dict. MBAT binding stores the
   random matrices here, keyed by ``"matrices"``.

   :returns: ``Dict[str, Any]``

Special methods
^^^^^^^^^^^^^^^

.. py:method:: Hypervector.__getitem__(key)
   :no-index:

   Index or slice a batch hypervector.

   .. code-block:: python

      batch = enc.generate(size=100)
      hv0   = batch[0]          # shape (10000,)
      first5 = batch[:5]        # shape (5, 10000)

.. py:method:: Hypervector.__len__()
   :no-index:

   Return the number of hypervectors in a batch (``shape[0]``).

.. py:method:: Hypervector.__repr__()
   :no-index:

   Human-readable summary: shape, dtype, backend.
