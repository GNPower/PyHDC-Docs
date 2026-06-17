How to Choose the Right Encoding
=================================

PyHDC provides 15 encoding classes. The decision tree and notes below narrow the choice for most use cases.

Quick decision guide
---------------------

Start here and follow the branches:

* **Need GPU / PyTorch integration?** → all encodings support both backends
* **Need binary output (1-bit elements)?**

  * Dense binary → :class:`~pyhdc.BSC` or :class:`~pyhdc.MAP_B`
  * Sparse binary (low density) → :class:`~pyhdc.BSDC_THIN` *(best default)* or
    :class:`~pyhdc.BSDC_S` / :class:`~pyhdc.BSDC_SEG` / :class:`~pyhdc.BSDC_CDT`

* **Need complex / phase-based values?** → :class:`~pyhdc.FHRR`
* **Need matrix-style binding?** → :class:`~pyhdc.VTB` or :class:`~pyhdc.MBAT`
* **Otherwise (continuous float vectors)?**

  * General purpose → :class:`~pyhdc.MAP_C` *(best default)*
  * Normalized bundling theory → :class:`~pyhdc.HRR`
  * Fixed-width integers → :class:`~pyhdc.MAP_I` or :class:`~pyhdc.MAP_I_Bits`

Full comparison table
----------------------

.. list-table::
   :header-rows: 1
   :widths: 12 10 10 16 16 10 10 16

   * - Encoding
     - Element type
     - Default dtype
     - Binding
     - Bundling
     - Similarity
     - Unbind
     - Notes
   * - ``MAP_C``
     - float [-1,1]
     - float32
     - ElementMultiply
     - Add + cut
     - Cosine
     - Yes
     - Best all-round default
   * - ``MAP_I``
     - int {-1,1}
     - int32
     - ElementMultiply
     - Add + cut
     - Cosine
     - Yes
     - Integer; needs bit generator
   * - ``MAP_I_Bits``
     - int (custom width)
     - int32
     - ElementMultiply
     - Add (clipped)
     - Cosine
     - Yes
     - ``mask`` sets bit width
   * - ``MAP_B``
     - binary {0,1}
     - int8
     - ElementMultiply
     - Add (clipped)
     - Cosine
     - Yes
     - Binary MAP
   * - ``HRR``
     - float (normal)
     - float32
     - CircularConv
     - Add + normalise
     - Cosine
     - Yes
     - Theoretically clean
   * - ``HRR_NoNorm``
     - float (normal)
     - float32
     - CircularConv
     - Add (no norm)
     - Cosine
     - Yes
     - Faster than HRR
   * - ``HRR_ConstNorm``
     - float (normal)
     - float32
     - CircularConv
     - Add / √M
     - Cosine
     - Yes
     - Constant-norm bundles
   * - ``FHRR``
     - angle [0, 2π]
     - float32
     - AngleAdd
     - AngleAdd
     - AngleDist
     - Yes
     - Phase/periodic signals
   * - ``VTB``
     - float (normal)
     - float32
     - VDTransform
     - Add + normalise
     - Cosine
     - Yes
     - Matrix derived from key
   * - ``MBAT``
     - float (normal)
     - float32
     - MatrixMult
     - Add + normalise
     - Cosine
     - Yes (+ metadata)
     - Random matrix; save ``get_metadata()``
   * - ``BSC``
     - binary {0,1}
     - int8
     - XOR
     - Majority vote
     - Hamming
     - Yes (exact)
     - Dense binary; XOR is exact inverse
   * - ``BSDC_CDT``
     - sparse {0,1}
     - int8
     - CDThinning
     - OR
     - Overlap
     - **No**
     - Context-dependent thinning
   * - ``BSDC_S``
     - sparse {0,1}
     - int8
     - CircShift
     - OR
     - Overlap
     - Yes
     - Shift-based; good for sequences
   * - ``BSDC_SEG``
     - sparse segmented
     - int8
     - SegShift
     - OR
     - Overlap
     - Yes
     - Per-segment shift
   * - ``BSDC_THIN``
     - sparse {0,1}
     - int8
     - CircShift
     - OR + thin
     - Overlap
     - Yes
     - Best sparse default (v1.1.0+)

Side-by-side example
---------------------

The encoding API is identical across families; only the constructor call
changes:

.. code-block:: python

   import pyhdc

   for EncClass in [pyhdc.MAP_C, pyhdc.HRR, pyhdc.BSC, pyhdc.BSDC_THIN]:
       enc = EncClass(dimension=10_000)
       a   = enc.generate()
       b   = enc.generate()
       c   = a.bind(b)
       print(f"{EncClass.__name__:12s}  "
             f"sim(a,a)={a.similarity(a):.2f}  "
             f"sim(a,bind(a,b))={a.similarity(c):.2f}")
