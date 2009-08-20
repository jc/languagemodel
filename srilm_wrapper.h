#ifndef SRILMWRAP_H
#define SRILMWRAP_H

#ifdef __cplusplus
extern "C" {
#else
    typedef struct Ngram Ngram; /* dummy type to stand in for class */
#endif

    Ngram* initLM(int order);
    void deleteLM(Ngram* ngram);
    unsigned getIndexForWord(const char* s);
    const char* getWordForIndex(unsigned i);
    int readLM(Ngram* ngram, const char* filename);
    int readVocab(const char* filename);
    float getWordProb(Ngram* ngram, unsigned word, unsigned* context);
    float wordProb(Ngram* ngram, unsigned w, unsigned* context);
    float getProb(Ngram* ngram, unsigned order, char* ngramstr);
    float getSentenceProb(Ngram* ngram, const char* sentence, unsigned length);
    unsigned corpusStats(Ngram* ngram, const char* filename, TextStats &stats);
    float getCorpusProb(Ngram* ngram, const char* filename);
    float getCorpusPpl(Ngram* ngram, const char* filename);
    int howManyNgrams(Ngram* ngram, unsigned order);
    float getUnigramProb(Ngram* ngram, const char* word);
    float getBigramProb(Ngram* ngram, const char* ngramstr);
    float getTrigramProb(Ngram* ngram, const char* ngramstr);
      
#ifdef __cplusplus
  }
#endif

#endif

