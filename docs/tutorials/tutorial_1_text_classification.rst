Tutorial 1: Encoding Text for Classification
=============================================

This tutorial builds a simple language classifier using PyHDC. The classifier
distinguishes Python keywords from common English nouns by encoding each word
as a hypervector and comparing it to class prototypes. It covers building a
character codebook, n-gram encoding, prototype construction, and similarity
classification.

**Prerequisites**: :doc:`../getting_started/quickstart`

----

What we are building
--------------------

Given an unknown word, we want to predict whether it is a Python keyword
(``if``, ``for``, ``class``, …) or an ordinary English noun (``cat``,
``house``, ``river``, …).

The HDC approach:

1. Build a codebook: one random hypervector per character of the alphabet.
2. Encode each word as a trigram hypervector (bind adjacent characters,
   bundle across all windows).
3. Build a class prototype for each category by bundling all its word vectors.
4. Classify a new word by comparing it to each prototype.

This is a one-pass algorithm: no gradient descent, no epochs.

----

Step 1: Set up the encoding and character codebook
---------------------------------------------------

.. code-block:: python

   import pyhdc
   import string

   # Use MAP_B with 10,000 dimensions
   enc = pyhdc.MAP_B(dimension=10_000)

   # One random hypervector per printable character
   alphabet = string.ascii_lowercase + string.digits + '_'
   char_hv  = {ch: enc.generate() for ch in alphabet}

Every character gets its own unique, random hypervector. Because the
hypervectors are drawn independently, any two characters are almost perfectly
orthogonal.

----

Step 2: Encode a word as an n-gram hypervector
-----------------------------------------------

A *trigram* is a window of three consecutive characters. We encode each
trigram by binding the three character vectors together (binding creates an
ordered record), then bundle all trigrams for the word into a single vector.

.. code-block:: python

   def encode_word(word, enc, char_hv, n=3):
       """Return a hypervector representing the n-gram profile of a word."""
       word = word.lower()
       if len(word) < n:
           # For short words, pad with a special placeholder
           word = word.ljust(n, '_')

       trigram_hvs = []
       for i in range(len(word) - n + 1):
           trigram = word[i:i + n]
           # Bind the three character vectors together (order matters)
           hv = char_hv[trigram[0]].bind(char_hv[trigram[1]]).bind(char_hv[trigram[2]])
           trigram_hvs.append(hv)

       # Bundle all trigrams into one word-level hypervector
       return pyhdc.bundle(*trigram_hvs)

Why binding? Because ``bind(a, bind(b, c))`` is different from
``bind(b, bind(a, c))``; binding is sensitive to order, so the encoded
hypervector carries positional information.  Bundling then aggregates all
windows, making the result insensitive to the exact number of trigrams.

Let's verify that two different words produce dissimilar hypervectors, and
that the same word produces the same hypervector:

.. code-block:: python

   hv_cat  = encode_word('cat',  enc, char_hv)
   hv_car  = encode_word('car',  enc, char_hv)
   hv_cat2 = encode_word('cat',  enc, char_hv)

   print(hv_cat.similarity(hv_cat2))  # 1.0   # deterministic
   print(hv_cat.similarity(hv_car))   # ~= 0.5  # one trigram in common ("ca")

----

Step 3: Build class prototypes
-------------------------------

A *class prototype* is the bundle of all word hypervectors that belong to
that class. The bundle is an approximation of the class as a whole. Queries
against it will return high similarity for words "in" the class.

.. code-block:: python

   python_keywords = [
       'false', 'none', 'true', 'and', 'as', 'assert', 'async', 'await',
       'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
       'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
       'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
       'while', 'with', 'yield',
   ]

   english_nouns = [
       'cat', 'dog', 'house', 'river', 'cloud', 'tree', 'book', 'chair',
       'table', 'stone', 'light', 'water', 'music', 'road', 'city', 'field',
       'door', 'window', 'flower', 'bridge', 'mountain', 'ocean', 'forest',
       'garden', 'street', 'candle', 'bottle', 'letter', 'mirror', 'paper',
   ]

   # Build one prototype per class
   kw_prototype   = pyhdc.bundle(*[encode_word(w, enc, char_hv) for w in python_keywords])
   noun_prototype = pyhdc.bundle(*[encode_word(w, enc, char_hv) for w in english_nouns])

----

Step 4: Classify a new word
----------------------------

To classify a query word, encode it and compare it to each prototype. The
class with the highest similarity wins.

.. code-block:: python

   def classify(word, enc, char_hv, prototypes):
       hv = encode_word(word, enc, char_hv)
       scores = {cls: hv.similarity(proto) for cls, proto in prototypes.items()}
       return max(scores, key=scores.get), scores

   prototypes = {'keyword': kw_prototype, 'noun': noun_prototype}

   for word in ['import', 'lamp', 'yield', 'stone', 'while', 'mirror']:
       pred, scores = classify(word, enc, char_hv, prototypes)
       print(f"{word:10s} -> {pred:10s}  "
             f"(keyword={scores['keyword']:+.3f}, noun={scores['noun']:+.3f})")

Expected output (values will vary slightly due to randomness in the codebook):

.. code-block:: text

   import     -> keyword     (keyword=+0.312, noun=-0.021)
   lamp       -> noun        (keyword=-0.018, noun=+0.287)
   yield      -> keyword     (keyword=+0.289, noun=+0.003)
   stone      -> noun        (keyword=-0.014, noun=+0.301)
   while      -> keyword     (keyword=+0.315, noun=-0.009)
   mirror     -> noun        (keyword=-0.022, noun=+0.294)

Vectorized classification with cross-similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loop above scores one query at a time. To score a whole test set in a
single call, stack the two prototypes into a ``(D, 2)`` batch and the encoded
test words into a ``(D, T)`` batch, then pass ``mode="cross"`` to
:meth:`~pyhdc.Hypervector.similarity`. The result is the full ``(2, T)`` matrix
of every prototype against every test word, and ``argmax`` down axis 0 picks the
winning class per column:

.. code-block:: python

   import numpy as np

   protos = pyhdc.stack([kw_prototype, noun_prototype])     # (D, 2)
   test_words = ['import', 'lamp', 'yield', 'stone']
   testhv = pyhdc.stack(
    [encode_word(w, enc, char_hv) for w in test_words]      # (D, T)
   )
   scores = protos.similarity(testhv, mode="cross")         # (2, T)
   labels = ['keyword', 'noun']
   for j, w in enumerate(test_words):
       print(w, labels[int(np.asarray(scores)[:, j].argmax())])

Column ``j`` of ``scores`` holds ``[kw_score, noun_score]`` for test word ``j``,
so this gives the same labels as the per-query loop with one matmul instead of a
Python loop over similarity calls.

----

Step 5: Measure accuracy
-------------------------

Let's hold out 20% of the data and measure accuracy on it.

.. code-block:: python

   import random

   random.seed(0)
   all_data = (
       [(w, 'keyword') for w in python_keywords] +
       [(w, 'noun')    for w in english_nouns]
   )
   random.shuffle(all_data)

   split   = int(0.8 * len(all_data))
   train   = all_data[:split]
   test    = all_data[split:]

   # Rebuild prototypes from training set only
   kw_proto_train   = pyhdc.bundle(*[encode_word(w, enc, char_hv)
                                     for w, label in train if label == 'keyword'])
   noun_proto_train = pyhdc.bundle(*[encode_word(w, enc, char_hv)
                                     for w, label in train if label == 'noun'])
   protos_train = {'keyword': kw_proto_train, 'noun': noun_proto_train}

   correct = sum(
       classify(w, enc, char_hv, protos_train)[0] == label
       for w, label in test
   )
   print(f"Accuracy: {correct}/{len(test)} = {correct/len(test):.0%}")

With ``dimension=10_000`` you should see accuracy of around 90-100% on this
small dataset.

----

Experiment: effect of dimension
---------------------------------

HDC accuracy improves with dimension. Try re-running with smaller dimensions:

.. code-block:: python

   for dim in [500, 1_000, 2_000, 5_000, 10_000]:
       enc_d    = pyhdc.MAP_B(dimension=dim)
       char_d   = {ch: enc_d.generate() for ch in alphabet}
       kw_p     = pyhdc.bundle(*[encode_word(w, enc_d, char_d) for w in python_keywords])
       noun_p   = pyhdc.bundle(*[encode_word(w, enc_d, char_d) for w in english_nouns])
       protos_d = {'keyword': kw_p, 'noun': noun_p}
       c = sum(classify(w, enc_d, char_d, protos_d)[0] == label for w, label in all_data)
       print(f"dim={dim:6d}: {c}/{len(all_data)} = {c/len(all_data):.0%}")

You will observe that accuracy rises sharply with dimension and plateaus
somewhere around 5,000-10,000.

----

Switching encoding families
-----------------------------

The encode/classify API is completely independent of the encoding family.
Replace ``MAP_B`` with ``HRR`` and nothing else changes:

.. code-block:: python

   enc_hrr  = pyhdc.HRR(dimension=10_000)
   char_hrr = {ch: enc_hrr.generate() for ch in alphabet}
   kw_hrr   = pyhdc.bundle(*[encode_word(w, enc_hrr, char_hrr) for w in python_keywords])
   noun_hrr = pyhdc.bundle(*[encode_word(w, enc_hrr, char_hrr) for w in english_nouns])
   # ... same classify() function works unchanged
    
HRR uses circular convolution and correlation internally, but the ``.bind()``,
``.bundle()``, and ``.similarity()`` methods have the same interface.

----

Where to go next
-----------------

* :doc:`tutorial_2_associative_memory` : store and retrieve key-value pairs
* :doc:`../how_to/choose_encoding` : when to use MAP_C vs. BSC vs. HRR
* :doc:`../user_manual/encodings_overview` : full comparison of all encoding families
