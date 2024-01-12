import sys
from conllu import parse
from conllu import parse_incr
from utils import loadVocabFile
import torch
import torch.nn as nn
from BILSTMConvolution import BiLSTMConvolutionTagger
from UDTagSet import UDTagSet

def decode(conlluFileName, model, dicoVocab, tagSet) :
    data_file = open(conlluFileName, "r", encoding="utf-8")
    for sentence in parse_incr(data_file):
        decodeSentence(sentence, model, dicoVocab, tagSet)
        print(sentence.serialize(), end="")

def decodeSentence(sentence, model, dicoVocab, tagSet) :
    x = []
    for token in sentence :
        form = token['form']
        if form not in dicoVocab :
            formCode = dicoVocab['<UNK>']
        else :
            formCode = dicoVocab[form]
        x.append(formCode)
    input_vec = torch.tensor(x)
    yprime, h =  model.forward(input_vec)
    for index, token in enumerate(sentence) :
        yhat = torch.argmax(yprime[index]).item()
        token['feats'] = tagSet.codeToTag(yhat)

def main():
    if len(sys.argv) < 4:
        print("Usage :", sys.argv[0], "model conlluFile vocabFile tagSetFile")
        sys.exit(1)


    modelFileName = sys.argv[1]
    conlluFileName = sys.argv[2]
    vocabFileName  = sys.argv[3]
    tagSetFile = sys.argv[4]

    model = torch.load(modelFileName)

    dicoVocab = loadVocabFile(vocabFileName)
    vocabSize = len(dicoVocab)

    tagSet = UDTagSet(tagSetFile, True)

    decode(conlluFileName, model, dicoVocab, tagSet)

if __name__ == '__main__':
    main()
