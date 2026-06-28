API Reference
=============

This section provides complete reference documentation for every public class,
function, and exception in PyHDC.

.. toctree::
   :maxdepth: 1

   pyhdc
   hypervector
   encoding_base
   encodings
   encoders
   generation
   components
   exceptions
   types

Pages
-----

:doc:`pyhdc`
   Top-level module: version info, ``TORCH_AVAILABLE``, convenience functions
   (``generate``, ``zeros``, ``bundle``, ``bind``, ``unbind``, ``similarity``), and type aliases.

:doc:`hypervector`
   The :class:`~pyhdc.Hypervector` class: all properties and methods.

:doc:`encoding_base`
   The :class:`~pyhdc.Encoding` abstract base class, ``EncodingSpec`` dataclass,
   and ``BackendManager``.

:doc:`encodings`
   All 15 encoding classes: MAP_C, MAP_I, MAP_I_Bits, MAP_B, HRR, HRR_NoNorm,
   HRR_ConstNorm, FHRR, VTB, MBAT, BSC, BSDC_CDT, BSDC_S, BSDC_SEG, BSDC_THIN.

:doc:`encoders`
   The :class:`~pyhdc.encoders.Encoder` base and all data encoders: codebook
   (``Empty``, ``Identity``, ``Random``, ``Level``, ``Thermometer``, ``Circular``)
   and functional (``Projection``, ``Sinusoid``, ``Density``, ``FractionalPower``).

:doc:`generation`
   The :class:`~pyhdc.generation.HDCGenerator` abstract base class,
   ``DefaultGenerator``, and all seven generator families.

:doc:`components`
   All functions in ``pyhdc.components``: binding, bundling, similarity, basis,
   quantization, elements, and thinning.

:doc:`exceptions`
   The ``HDCException`` hierarchy and when each exception is raised.

:doc:`types`
   Type aliases: ``Backend``, ``ArrayLike``, ``Device``, ``GeneratorOutputType``.
