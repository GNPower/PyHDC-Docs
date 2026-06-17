How to Handle PyHDC Exceptions
================================

PyHDC raises specific exceptions for different error conditions. The exception
hierarchy, trigger conditions, and handling patterns are documented below.

Exception hierarchy
--------------------

.. code-block:: text

   Exception
   └── HDCException
       ├── DimensionsNotMatchingError
       ├── DtypesNotMatchingError
       └── GeneratorNotSupportedError

All PyHDC-specific errors inherit from ``HDCException``, so you can catch the
base class to handle any library error.

When each exception is raised
-------------------------------

**DimensionsNotMatchingError**

Raised when you try to bind or bundle two hypervectors with different
dimensions:

.. code-block:: python

   import pyhdc

   a = pyhdc.MAP_C(dimension=10_000).generate()
   b = pyhdc.MAP_C(dimension=5_000).generate()

   try:
       a.bind(b)
   except pyhdc.DimensionsNotMatchingError as e:
       print(f"Dimension mismatch: {e}")

Fix: ensure all hypervectors in an operation use the same ``dimension``.

**DtypesNotMatchingError**

Raised when two hypervectors have incompatible data types for an operation:

.. code-block:: python

   from pyhdc import DtypesNotMatchingError

   try:
       result = a.bind(b)
   except DtypesNotMatchingError as e:
       print(f"Dtype mismatch: {e}")

**GeneratorNotSupportedError**

Raised when a generator is paired with an encoding that requires a different
output type (bits, words, or floats):

.. code-block:: python

   from pyhdc.generation import CommonLFSRGenerators

   gen = CommonLFSRGenerators.fibonacci_16(seed=1)
   enc = pyhdc.MAP_C(dimension=10_000, generator=gen)   # MAP_C needs floats

   try:
       enc.generate()
   except pyhdc.GeneratorNotSupportedError:
       # Fall back to default generator
       enc = pyhdc.MAP_C(dimension=10_000)
       hv  = enc.generate()

**NotImplementedError (not an HDCException)**

Calling ``.unbind()`` on an encoding that does not support it (e.g.,
``BSDC_CDT``) raises Python's built-in ``NotImplementedError``:

.. code-block:: python

   enc = pyhdc.BSDC_CDT(dimension=10_000)
   a, b = enc.generate(), enc.generate()
   bound = a.bind(b)

   try:
       bound.unbind(b)
   except NotImplementedError:
       print("BSDC_CDT does not support unbinding")

**ValueError (not an HDCException)**

Backend mismatch raises ``ValueError``, not ``HDCException``:

.. code-block:: python

   hv_np    = pyhdc.MAP_C(dimension=10_000).generate()
   hv_torch = pyhdc.MAP_C(dimension=10_000, backend="torch").generate()

   try:
       hv_np.similarity(hv_torch)
   except ValueError as e:
       print(f"Backend mismatch: {e}")
   except pyhdc.HDCException as e:
       print(f"HDC error: {e}")

Catching all PyHDC errors
--------------------------

.. code-block:: python

   try:
       result = enc.bundle(hv1, hv2)
   except pyhdc.HDCException as e:
       # catches DimensionsNotMatchingError, DtypesNotMatchingError,
       # and GeneratorNotSupportedError subclasses
       print(f"PyHDC error: {type(e).__name__}: {e}")
   except (ValueError, NotImplementedError) as e:
       # backend mismatch or unsupported operation
       print(f"Operation error: {e}")
