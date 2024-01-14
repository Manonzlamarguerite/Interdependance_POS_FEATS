import sys
from conllu import parse
from conllu import parse_incr
from utils import loadVocabFile
import torch
import torch.nn as nn
from GRUNetMultiLabel import GRUNetMultiLabel
from UDTagSet import UDTagSet
import numpy as np

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
    yprime, predicted =  model.forward(input_vec)
    for index, token in enumerate(sentence) :
        proba = yprime[index].detach().numpy()
        best_index = []
        tagSet.createFeatList()
        # print(tagSet.list_feat)
        for feat in tagSet.list_feat:
            index_f = tagSet.featToIndice(feat)
            proba_feat = []
            for i in index_f:
                proba_feat.append(proba[i])
            best_i = np.argmax(proba_feat)
            if proba_feat[best_i] > 0.5:
                best_index.append(index_f[best_i])

        yhat = []
        for i in range(len(proba)):
            if i in best_index:
                yhat.append(1)
            else:
                yhat.append(0)
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
