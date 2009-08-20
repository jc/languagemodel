%module srilm_wrapper
%include "typemaps.i"
%import carrays.i
%array_functions(unsigned int, intArray)
%{
/* Include the header files etc. here */
#include "Ngram.h"
#include "Vocab.h"
#include "Prob.h"
#include "srilm_wrapper.h"
%}

%include srilm_wrapper.h

