Type Aliases
=============

.. module:: pyhdc.types

PyHDC uses typed aliases throughout its API. They are defined in
``pyhdc.types`` and re-exported from ``pyhdc`` for convenience.

Core types
----------

.. py:data:: Backend

   .. code-block:: python

      Backend = Literal["numpy", "torch"]

   Specifies which array backend a hypervector or encoding uses.

.. py:data:: ArrayLike

   .. code-block:: python

      ArrayLike = Union[numpy.ndarray, torch.Tensor]

   The raw array types that PyHDC can operate on. NumPy arrays are always
   supported; PyTorch tensors require ``TORCH_AVAILABLE = True``.

.. py:data:: Device

   .. code-block:: python

      Device = Union[str, torch.device]

   A device specification for PyTorch. Common values: ``"cpu"``,
   ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``. Ignored when ``backend="numpy"``.

.. py:data:: GeneratorOutputType

   .. code-block:: python

      GeneratorOutputType = Literal["bits", "words", "floats"]

   The type of values a generator produces. Each encoding family requires
   a specific output type from its generator. See the compatibility table in
   :doc:`../user_manual/generators`.

Composite types
---------------

.. py:data:: HypervectorLike

   .. code-block:: python

      HypervectorLike = Union[Hypervector, List[Hypervector]]

   Accepted anywhere a single hypervector or a list of hypervectors is valid.

.. py:data:: HypervectorInput

   .. code-block:: python

      HypervectorInput = Union[
          ArrayLike,
          Hypervector,
          List[Hypervector],
          List[List[Hypervector]],
      ]

   The broadest input type for encoding methods like ``bundle`` and ``bind``.
   Supports raw arrays, single hypervectors, flat lists, and nested lists
   for batched operations.

.. py:data:: SimilarityResult

   .. code-block:: python

      SimilarityResult = Union[float, ArrayLike, List[Union[float, ArrayLike]]]

   The return type of similarity functions. A scalar float for single-pair
   comparisons; an array for batched comparisons; a list for list inputs.

.. py:data:: SequenceType

   .. code-block:: python

      SequenceType = Union[List[int], List[float], ArrayLike]

   Used internally by the recovery module (not yet public).

Metadata types
--------------

.. py:data:: OperationMetadata

   .. code-block:: python

      OperationMetadata = Dict[str, Any]

   The metadata dict attached to a :class:`~pyhdc.Hypervector`. Currently
   used by MBAT to store binding matrices under the key ``"matrices"``.

.. py:data:: OperationResult

   .. code-block:: python

      OperationResult = Union[ArrayLike, Tuple[ArrayLike, OperationMetadata]]

   The return type of raw component functions. Most return a plain array;
   MBAT binding returns ``(array, metadata_dict)``.
