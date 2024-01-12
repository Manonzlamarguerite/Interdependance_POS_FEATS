import sys
from conllu import parse
from conllu import parse_incr
from torch.nn.utils.rnn import pad_sequence

from utils import loadVocabFile
import torch
import torch.nn as nn

from UDTagSet import UDTagSet

def decode(conlluFileName, model, dicoVocab, tagSet) :
    data_file = open(conlluFileName, "r", encoding="utf-8")
    for sentence in parse_incr(data_file):
        decodeSentence(sentence, model, dicoVocab, tagSet)
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
            Char.append(dicoCharVocab[char])
        c.append(Char)

    input_vec = torch.tensor(x)
    input_char = pad_sequence([torch.tensor(seq) for seq in c], batch_first=True, padding_value=0)
    yprime, h = model.forward(input_vec, input_char)
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

    vocabCharFileName = sys.argv[5]
    model = torch.load(modelFileName)
    dicoCharVocab = loadVocabFile(vocabCharFileName)
    dicoVocab = loadVocabFile(vocabFileName)
    vocabSize = len(dicoVocab)

    tagSet = UDTagSet(tagSetFile, True)

    decode(conlluFileName, model, dicoVocab, dicoCharVocab, tagSet)
if __name__ == '__main__':
    main()
