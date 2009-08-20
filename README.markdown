Python SWIG wrapper for the SRI Language Modeling (SRILM) Toolkit
=================================================================

Use your SRILM language model's from Python. The code is based off
[Nitin Madnani's][1] wrappers.

Requirements
------------

* GNU make
* Simplified Wrapper and Interface Generator (SWIG)
* Python 2.4+
* SRI Language Modeling Toolkit

Installation
------------

1. Copy the appropriate makefile to Makefile.
2. Modify the environmental variables at the top of Makefile.
3. run `make`

Usage
-----
     
     >>> from srilm import LanguageModel
     >>> lm = LanguageModel(lm_directory="/Users/james/languagemodels", vocab_file="gigaword.vocab", lm_file="gigaword.lm", n=3)
     >>> lm.trigram("new", "york", "times")
     -1.0087490081787109
     >>> lm.trigram("new", "york", "city")
     -0.8258976936340332
     >>>

See `srilm.py` for more details.



[1]: http://www.umiacs.umd.edu/~nmadnani
