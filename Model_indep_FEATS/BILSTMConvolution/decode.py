import sys
from conllu import parse
from conllu import parse_incr
from utils import loadVocabFile
import torch
import torch.nn as nn
from BILSTMConvolution import BiLSTMConvolutionTagger
from UDTagSet import UDTagSet
from torch._C._nn import pad_sequence


def decode(conlluFileName, model, dicoVocab, tagSet,dicoCharVocab) :
    data_file = open(conlluFileName, "r", encoding="utf-8")
    for sentence in parse_incr(data_file):
        decodeSentence(sentence, model, dicoVocab, tagSet, dicoCharVocab)
        print(sentence.serialize(), end="")

def decodeSentence(sentence, model, dicoVocab, dicoCharVocab,tagSet) :
    x = []
    c = []
    for token in sentence:
        Char = []
        form = token['form']
        if form not in dicoVocab:
            formCode = dicoVocab['<UNK>']
        else:
            formCode = dicoVocab[form]
        x.append(formCode)
        for char in form:
            if char in dicoCharVocab.keys():
                Char.append(dicoCharVocab[char])
            else:
                Char.append(dicoCharVocab["<UNK>"])
        c.append(Char)

    input_vec = torch.tensor(x)
    input_char = pad_sequence([torch.tensor(seq) for seq in c], batch_first=True, padding_value=0)
    yprime, h = model.forward(input_vec, input_char)
    for index, token in enumerate(sentence) :
        yhat = torch.argmax(yprime[index]).item()
        token['feats'] = tagSet.codeToTag(yhat)

def main():
    if len(sys.argv) < 6:
        print("Usage :", sys.argv[0], "model conlluFile vocabFile tagSetFile vocabCharFile")
        sys.exit(1)


    modelFileName = sys.argv[1]
    conlluFileName = sys.argv[2]
    vocabFileName  = sys.argv[3]
    tagSetFile = sys.argv[4]
    vocabCharFile = sys.argv[5]

    model = torch.load(modelFileName)

    dicoVocab = loadVocabFile(vocabFileName)
    dicoCharVocab = loadVocabFile(vocabCharFile)
    vocabSize = len(dicoVocab)

    tagSet = UDTagSet(tagSetFile, True)

    decode(conlluFileName, model, dicoVocab, dicoCharVocab,tagSet)

if __name__ == '__main__':
    main()
