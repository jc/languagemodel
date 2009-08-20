/* This is based off the original srilm wrapper by Nitin Madnani.
   <http://www.umiacs.umd.edu/~nmadnani/>

   Modifications:
   * compiles on OS X
   * assume open vocab language model and load vocab file (readVocab)
   * allows for using indices rather than strings (see wordProb)
   * get probability for string provided given an n-gram order (getProb)
   
   TODO: Can we make wordProb nicer? silly for loop for duplicating input
*/

#include "Prob.h"
#include "Ngram.h"
#include "Vocab.h"

#include "srilm_wrapper.h"
#include <cstdio>
#include <cstring>
#include <cmath>

/* strdupa */
#ifndef strdupa
#define strdupa(s)                                                            \
  (__extension__                                                              \
    ({                                                                        \
      __const char *__old = (s);                                              \
      size_t __len = strlen (__old) + 1;                                      \
      char *__new = (char *) __builtin_alloca (__len);                        \
      (char *) memcpy (__new, __old, __len);                                  \
    }))
#endif


const float BIGNEG = -99;
Vocab *swig_srilm_vocab;

// Initialize the ngram model
Ngram* initLM(int order) {
    return new Ngram(*swig_srilm_vocab, order);
}

// Delete the ngram model
void deleteLM(Ngram* ngram) {
  delete swig_srilm_vocab;
  delete ngram;
}

// Get index for given string
unsigned getIndexForWord(const char *s) {
  unsigned ans;
  ans = swig_srilm_vocab->getIndex((VocabString)s);
  if(ans == Vocab_None) {
      ans = swig_srilm_vocab->unkIndex();
  }
  return ans;
}

// Get the word for a given index
const char* getWordForIndex(unsigned i) {
  return swig_srilm_vocab->getWord((VocabIndex)i);
}

// Read in an LM file into the model
int readLM(Ngram* ngram, const char* filename) {
    File file(filename, "r");
    if(!file) {
        fprintf(stderr,"Error:: Could not open file %s\n", filename);
        return 0;
    }
    else 
        return ngram->read(file, 0);
}

//Read in an Vocab file into the model
int readVocab(const char* filename) {
    swig_srilm_vocab = new Vocab;
    swig_srilm_vocab->unkIsWord() = true;
    File file(filename, "r");
    if (!file) {
        fprintf(stderr, "Error:: Could not open file %s\n", filename);
        return 0;
    } else {
        return swig_srilm_vocab->read(file);
    }
}

// Get word probability
float getWordProb(Ngram* ngram, unsigned w, unsigned* context) {
    return (float)ngram->wordProb(w, context);
}

// I cannot pass Vocab_None in through swig so I have to duplicate
// the array and then assign the final element Vocab_None
// glibc complained lots when I tried to do it on context directly
float wordProb(Ngram* ngram, unsigned w, unsigned* context) {
    unsigned n = sizeof(context);
    unsigned indices[n];
    for(unsigned i = 0; i < n-1; i++)
        indices[i] = context[i];
    indices[n-1] = Vocab_None;
        
    return (float)ngram->wordProb(w, indices);
}

// get order-gram probability 
float getProb(Ngram* ngram, unsigned order, char* ngramstr) {
    const char* words[order];
    unsigned indices[order];
    unsigned numparsed;
    float ans;

    numparsed = Vocab::parseWords(ngramstr, (VocabString *)words, order);
    if (numparsed != order) {
        fprintf(stderr, "Error: Given ngram is not of order %d\n", order);
        return -1;
    }
    
    swig_srilm_vocab->getIndices((VocabString *)words, (VocabIndex *)indices, order);
    if (order == 1 && indices[0] == Vocab_None)
        indices[0] = swig_srilm_vocab->unkIndex();
    unsigned hist[order];
    for(unsigned i=0; i<order-1; i++) {
        hist[i] = indices[order-2-i];
    }
    hist[order-1] = Vocab_None;
    ans = getWordProb(ngram, indices[order-1], hist);
    if(ans == LogP_Zero) 
        return BIGNEG;

    return ans;
}

// Sentence Probability
float getSentenceProb(Ngram* ngram, const char* sentence, unsigned length) {
    float ans;
    const char* words[length];
    unsigned numparsed;
    TextStats stats;

    char* scp;
    //Create a copy of the input string to be safe
    scp = strdupa(sentence);

    numparsed = Vocab::parseWords(scp, (VocabString *)words, 15);
    if(numparsed != length) {
        fprintf(stderr, "Error: Number of words in sentence does not match given length.\n");
        return -1;
    }
    
    ans = ngram->sentenceProb(words, stats);
    if (ans == LogP_Zero) 
        return BIGNEG;

    return ans;
}

unsigned corpusStats(Ngram* ngram, const char* filename, TextStats &stats) {
    File corpus(filename, "r");

    if(!corpus) {
        fprintf(stderr,"Error:: Could not open file %s\n", filename);
        return 1;
    }
    else 
        ngram->pplFile(corpus, stats, 0);
        return 0;
}

float getCorpusProb(Ngram* ngram, const char* filename) {
    TextStats stats;
    if(!corpusStats(ngram, filename, stats))
        return stats.prob;
}

float getCorpusPpl(Ngram* ngram, const char* filename) {
    TextStats stats;
    float ans;
    
    if(!corpusStats(ngram, filename, stats)) {
        int denom = stats.numWords - stats.numOOVs - stats.zeroProbs + stats.numSentences;
        if (denom > 0) {
            ans = LogPtoPPL(stats.prob / denom);
        }
        else {
            ans = -1.0;
        }
        return ans;        
    }
}

// How many ngrams are in the model
int howManyNgrams(Ngram* ngram, unsigned order) {
  return ngram->numNgrams(order);
}

// Get trigram probability
float getTrigramProb(Ngram* ngram, const char* ngramstr) {
    const char* words[6];
    unsigned indices[3];
    unsigned numparsed;
    char* scp;
    float ans;

    // Duplicate 
    scp = strdupa(ngramstr);

    numparsed = Vocab::parseWords(scp, (VocabString *)words, 6);
    if(numparsed != 3) {
        fprintf(stderr, "Error: Given ngram is not a trigram.\n");
        return 0;
    }

    swig_srilm_vocab->addWords((VocabString *)words, (VocabIndex *)indices, 3);

    unsigned hist[3] = {indices[1], indices[0], Vocab_None};

    ans = getWordProb(ngram, indices[2], hist);

    if(ans == LogP_Zero) 
        return BIGNEG;

    return ans;
}


// Get bigram probability
float getBigramProb(Ngram* ngram, const char* ngramstr) {
    const char* words[2];
    unsigned indices[2];
    unsigned numparsed;
    char* scp;
    float ans;

    // Create a copy of the input string to be safe
    scp = strdupa(ngramstr);

    // Parse the bigram into the words
    numparsed = Vocab::parseWords(scp, (VocabString *)words, 2);
    if(numparsed != 2) {
        fprintf(stderr, "Error: Given ngram is not a bigram.\n");
        return -1;
    }

    // Add the words to the vocabulary
    swig_srilm_vocab->addWords((VocabString *)words, (VocabIndex *)indices, 2);

    // Fill the history array
    unsigned hist[2] = {indices[0], Vocab_None};

    // Compute the bigram probability
    ans = getWordProb(ngram, indices[1], hist);

    // Return the representation of log(0) if needed
    if(ans == LogP_Zero) 
        return BIGNEG;

    return ans;
}

// Get unigram probability
float getUnigramProb(Ngram* ngram, const char* word) {
    unsigned index;
    float ans;

    // fill the history array the empty token
    unsigned hist[1] = {Vocab_None};

    // get the index for this word
    index = getIndexForWord(word);

    // Compute word probability
    ans = getWordProb(ngram, index, hist);

    // If the probability is zero, return the constant representing
    // log(0). 
    if(ans == LogP_Zero) 
        return BIGNEG;

    return ans;
}
