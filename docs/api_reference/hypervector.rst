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
     - Shape of the underlying array (e.g., ``(10000,)`` or ``(10000, 100)``)
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

.. py:method:: Hypervector.similarity(other=None)
   :no-index:

   Compute similarity to another hypervector (or batch), or within a single
   batch when ``other`` is omitted.

   Delegates to the encoding's similarity function. Returns values in [-1, 1].

   :param other: A :class:`Hypervector` of the same encoding and backend. If
                 omitted, ``self`` must be a ``(D, N)`` batch and the similarity
                 of column 0 against each remaining column is returned.
   :returns: ``float``, ``ndarray``, or ``Tensor`` depending on input shapes.
             See :ref:`batched calling conventions <similarity-batched>`.

Batch selection
^^^^^^^^^^^^^^^

.. py:method:: Hypervector.select(indices)
   :no-index:

   Select hypervectors (columns) from a ``(D, N)`` batch by index along the
   batch axis. Returns a ``(D, len(indices))`` batch. Works on NumPy and PyTorch
   (list or array indices are accepted).

   .. code-block:: python

      codebook = enc.generate(size=(10_000, 100))   # (10000, 100)
      subset   = codebook.select([0, 2, 4])         # (10000, 3)

Bundle and bind
^^^^^^^^^^^^^^^

.. py:method:: Hypervector.bundle(*others, axis=None, batch_dim=None)
   :no-index:

   Bundle this hypervector with one or more others. A batched operand is reduced
   automatically, ``axis=`` selects which batch axis to collapse.

   :param others: Additional :class:`Hypervector` objects.
   :param axis: Batch axis (or tuple of axes) to reduce. Axis 0 cannot be reduced.
   :param batch_dim: Deprecated as of 2.1.0, emits ``DeprecationWarning`` and
                     will be removed in a future release. Pass a batched operand or use ``axis=``.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.bind(*others, batch_dim=None)
   :no-index:

   Bind this hypervector with one or more others. Batched operands are handled
   automatically: element-wise binders broadcast, others are applied per column.

   :param others: Additional :class:`Hypervector` objects.
   :param batch_dim: Deprecated as of 2.1.0, emits ``DeprecationWarning`` and
                     will be removed in a future release. Pass a batched operand instead.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.unbind(*others, batch_dim=None)
   :no-index:

   Unbind to recover a component. Batched operands are handled automatically.

   :param batch_dim: Deprecated as of 2.1.0, emits ``DeprecationWarning`` and
                     will be removed in a future release. Pass a batched operand instead.
   :raises NotImplementedError: For encodings that do not support unbinding
                                (e.g., ``BSDC_CDT``).
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.thin()
   :no-index:

   Apply the encoding's thinning operation (sparse binary encodings only).
   For encodings that do not thin, returns ``self`` unchanged.

   :returns: :class:`Hypervector`

Unary operations
^^^^^^^^^^^^^^^^

These four operate dimension-first. Axis 0 is always the dimension ``D``, and
each operation broadcasts over the trailing batch axes of a ``(D, N)`` or
``(D, N, M)`` input. Each delegates to the encoding (``self._encoding.<op>``)
and returns a new :class:`Hypervector`. Whether an operation is defined depends
on the encoding family. The per-family behavior is listed below and resolved
through the encoding's ``EncodingSpec``.

.. py:method:: Hypervector.permute(shift=1)
   :no-index:

   Cyclic-shift permutation along axis 0 (the dimension). A positive ``shift``
   rolls coordinates forward, a negative ``shift`` rolls them back and is the
   exact inverse of the positive shift. Every encoding defines ``permute``: when
   an encoding does not supply its own ``permute_fn``, the shared
   ``CyclicShift`` component is used, so all families support it by default.

   :param shift: Integer number of positions to roll. Default ``1``.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.inverse()
   :no-index:

   The binding inverse: the element that unbinds what binding produced. The
   mechanism is family-specific (``IdentityInverse`` for self-inverse binding,
   ``ReverseInverse`` for circular convolution, ``PhaseNegate`` for FHRR).

   :raises NotImplementedError: For encodings whose ``EncodingSpec`` leaves
                                ``inverse_fn`` at its default: ``MAP_C``,
                                ``VTB``, ``MBAT``, ``BSDC_CDT``, ``BSDC_S``,
                                ``BSDC_SEG``, and ``BSDC_THIN``.
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.normalize()
   :no-index:

   Normalize to the entry distribution for the encoding. The mechanism is family-specific
   (``SignNormalize`` to bipolar ``{-1, 0, +1}`` for MAP, ``L2Normalize`` to
   unit L2 length along axis 0 for HRR/VTB/MBAT, ``WrapPhase`` to ``[-pi, pi)``
   for FHRR).

   :raises NotImplementedError: For encodings whose ``EncodingSpec`` leaves
                                ``normalize_fn`` at its default: ``BSC`` and all
                                four BSDC variants (``BSDC_CDT``, ``BSDC_S``,
                                ``BSDC_SEG``, ``BSDC_THIN``).
   :returns: :class:`Hypervector`

.. py:method:: Hypervector.negative()
   :no-index:

   The bundling (additive) inverse: element-wise negation for the additive
   families (``Negate``).

   :raises NotImplementedError: For encodings whose ``EncodingSpec`` leaves
                                ``negative_fn`` at its default: ``FHRR``,
                                ``BSC``, and all four BSDC variants
                                (``BSDC_CDT``, ``BSDC_S``, ``BSDC_SEG``,
                                ``BSDC_THIN``).
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

      batch = enc.generate(size=(10_000, 100))   # (10000, 100)
      hv0   = batch[:, 0]        # shape (10000,): first hypervector (column 0)
      first5 = batch[:, :5]      # shape (10000, 5): first five hypervectors

Operators route through the encoding, so each one raises per family exactly as
the underlying method does. For ``+``, ``*``, and ``/`` a non-:class:`Hypervector`
right-hand operand returns ``NotImplemented``, which Python turns into a
``TypeError``. There are no reflected dunders, so ``other + hv`` with a
non-hypervector ``other`` also raises ``TypeError``.

.. code-block:: python

   a = enc.generate(size=10_000)
   b = enc.generate(size=10_000)

   bundled = a + b       # a.bundle(b)
   bound   = a * b       # a.bind(b)
   recov   = bound / b   # bound.unbind(b)
   inv     = ~a          # a.inverse()
   rolled  = a >> 3      # a.permute(shift=3)
   back    = rolled << 3 # a.permute(shift=-3), inverse of >> 3

.. py:method:: Hypervector.__add__(other)
   :no-index:

   Bundle. Routes to ``self.bundle(other)``. ``other`` must be a
   :class:`Hypervector`, any other type returns ``NotImplemented``. Defined for
   all encodings.

.. py:method:: Hypervector.__mul__(other)
   :no-index:

   Bind. Routes to ``self.bind(other)``. ``other`` must be a
   :class:`Hypervector`, any other type returns ``NotImplemented``. Defined for
   all encodings. On a batched (``ndim > 1``) operand the non-element-wise
   binders (HRR convolution, shifting, matrix, VTB, BSDC_CDT) are applied per
   column, returning one batched result.

.. py:method:: Hypervector.__truediv__(other)
   :no-index:

   Unbind. Routes to ``self.unbind(other)``. ``other`` must be a
   :class:`Hypervector`, any other type returns ``NotImplemented``.

   :raises NotImplementedError: For ``BSDC_CDT`` (its ``unbinding_fn`` is the
                                default raising stub).

.. py:method:: Hypervector.__invert__()
   :no-index:

   Inverse. Routes to ``self.inverse()``.

   :raises NotImplementedError: For ``MAP_C``, ``VTB``, ``MBAT``, ``BSDC_CDT``,
                                ``BSDC_S``, ``BSDC_SEG``, and ``BSDC_THIN`` (see
                                :py:meth:`Hypervector.inverse`).

.. py:method:: Hypervector.__rshift__(shift)
   :no-index:

   Permute forward. Routes to ``self.permute(shift=int(shift))``. ``shift`` must
   be integral (Python ``int`` or any ``numbers.Integral`` such as a NumPy
   integer) and must not be a ``bool``; a ``bool``, float, or other type returns
   ``NotImplemented`` (Python raises ``TypeError``). Defined for all encodings.

.. py:method:: Hypervector.__lshift__(shift)
   :no-index:

   Permute backward. Routes to ``self.permute(shift=-int(shift))`` and is the
   exact inverse of ``>>`` by the same amount. Same ``shift`` rule as
   :py:meth:`Hypervector.__rshift__`: integral and not ``bool``, else
   ``NotImplemented``. Defined for all encodings.

.. py:method:: Hypervector.__len__()
   :no-index:

   Return ``shape[0]``, which is the hypervector dimension ``D`` (axis 0), not
   the batch count, since batches are dimension-first ``(D, N)``.

.. py:method:: Hypervector.__repr__()
   :no-index:

   Human-readable summary: shape, dtype, backend.
