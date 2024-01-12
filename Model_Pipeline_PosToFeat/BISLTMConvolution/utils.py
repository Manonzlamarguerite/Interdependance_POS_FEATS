


def conlluPosTags():
    return {
    "ADJ"   : 0,
    "ADP"   : 1,
    "ADV"   : 2,
    "AUX"   : 3,
    "CCONJ" : 4,
    "DET"   : 5,
    "NOUN"  : 6,
    "NUM"   : 7,
    "PRON"  : 8,
    "PROPN" : 9,
    "PUNCT" : 10,
    "SCONJ" : 11,
    "SYM"   : 12,
    "VERB"  : 13,
    "INTJ"  : 14,
    "X"     : 15,
    "_"     : 16
    }

def loadVocabFile(fileName):
    dicoVocab = {}
    f = open(fileName, 'r')
    for line in f:
        line = line.rstrip()
        (code, form) = line.split()
        dicoVocab[form] = int(code)
    return dicoVocab
