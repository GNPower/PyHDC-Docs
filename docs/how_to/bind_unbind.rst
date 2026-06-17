How to Bind and Unbind Key-Value Pairs
=======================================

Binding creates an association between two hypervectors; unbinding inverts
it to recover one component given the other.

Single pair bind and unbind
-----------------------------

.. code-block:: python

   import pyhdc

   enc   = pyhdc.MAP_C(dimension=10_000)
   key   = enc.generate()
   value = enc.generate()

   bound     = key.bind(value)
   recovered = bound.unbind(key)

   print(recovered.similarity(value))   # ~= 1.0

The recovered vector is not identical to ``value``; it carries noise from
the finite dimension. Use it as a query against a codebook to identify the
closest match.

Building a multi-field record
-------------------------------

Bundle multiple bindings to create a record with several fields:

.. code-block:: python

   roles   = {r: enc.generate() for r in ['name', 'colour', 'shape']}
   names   = {n: enc.generate() for n in ['Alice', 'Bob']}
   colours = {c: enc.generate() for c in ['red', 'blue', 'green']}
   shapes  = {s: enc.generate() for s in ['circle', 'square']}

   alice_record = pyhdc.bundle(
       roles['name'].bind(names['Alice']),
       roles['colour'].bind(colours['red']),
       roles['shape'].bind(shapes['circle']),
   )

Querying a field
-----------------

.. code-block:: python

   def query_field(record, role_hv, codebook):
       result = record.unbind(role_hv)
       return max(codebook, key=lambda k: result.similarity(codebook[k]))

   print(query_field(alice_record, roles['colour'], colours))   # red
   print(query_field(alice_record, roles['shape'],  shapes))    # circle

Encodings that support unbinding
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Encoding
     - Unbind
     - Notes
   * - MAP_C, MAP_I, MAP_I_Bits, MAP_B
     - Yes
     - Element-wise multiply is self-inverse: ``bind(bind(a,b), b) ≈ a``
   * - HRR family, FHRR
     - Yes
     - Circular correlation inverts circular convolution
   * - VTB
     - Yes
     - Transpose of the transformation matrix
   * - MBAT
     - Yes (needs metadata)
     - See note on metadata below
   * - BSC
     - Yes (exact)
     - XOR is exactly self-inverse: ``a XOR b XOR b == a``
   * - BSDC_S, BSDC_SEG, BSDC_THIN
     - Yes
     - Inverse circular shift
   * - BSDC_CDT
     - No
     - ``unbind()`` raises ``NotImplementedError``

MBAT and metadata
------------------

MBAT binding uses a random matrix that must be stored for later inversion.
The matrix is returned in the result's metadata dictionary:

.. code-block:: python

   enc    = pyhdc.MBAT(dimension=1_000)
   key    = enc.generate()
   value  = enc.generate()

   bound  = key.bind(value)
   meta   = bound.get_metadata()   # contains 'matrices' key

   # Unbind using the stored matrices
   recovered = bound.unbind(key)
   print(recovered.similarity(value))   # ~= 1.0

When unbind is not available
------------------------------

If you are using ``BSDC_CDT`` or another encoding that does not support
unbinding, query via similarity search instead:

.. code-block:: python

   # For BSDC_CDT: use nearest-neighbour search on the value codebook
   result = bound.unbind(key)   # NotImplementedError for BSDC_CDT

   # Alternative: compare the bound record directly against all possible bound pairs
   candidates = {v_name: key.bind(v_hv) for v_name, v_hv in values.items()}
   best = max(candidates, key=lambda n: bound.similarity(candidates[n]))
