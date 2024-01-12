import sys
from conllu import parse
from conllu import parse_incr
from utils import conlluPosTags
import torch
import torch.nn as nn
from GRUNet import GRUNet

def extractVocab(conlluFileName):
    dicoVocab = {}
    data_file = open(conlluFileName, "r", encoding="utf-8")
    vocabSize = 0

    for sentence in parse_incr(data_file):
        for token in sentence :
            form = token['form']
            # On ajout l'information des traits morphologiques
            feats = token["feats"]
            if feats != None:
                for feat in feats.keys():
                    form += "|"+feat+"="+feats[feat]
            if form not in dicoVocab:
                dicoVocab[form] = vocabSize
                vocabSize += 1
    data_file.close()
    return dicoVocab

def prepareData(conlluFileName, dicoVocab, dicoUpos) :
    data_file = open(conlluFileName, "r", encoding="utf-8")
    LX = []
    LY = []
    for sentence in parse_incr(data_file):
        X = []
        Y = []
        for token in sentence :
            form = token['form']
            # On ajout l'information des traits morphologiques
            feats = token["feats"]
            if feats != None:
                for feat in feats.keys():
                    form += "|"+feat+"="+feats[feat]
            if form not in dicoVocab :
                formCode = dicoVocab['<UNK>']
            else :
                formCode = dicoVocab[form]
            X.append(formCode)
            upos = token['upos']
            uposCode = dicoUpos[upos]
            Y.append(uposCode)
        LX.append(X)
        LY.append(Y)
    return LX, LY

def train(LX, LY, model, nb_iter):
    optim = torch.optim.Adam(model.parameters())
    lossFct = torch.nn.CrossEntropyLoss()
    for iteration in range(nb_iter):
        total_loss = 0
        print("iteration : ", iteration + 1, file=sys.stderr)
        n = 0
        for (x,y) in zip(LX,LY):
            if n % 100 == 0 :
                print(".", end="", flush=True, file=sys.stderr)
            input_vec = torch.tensor(x)
            output_vec = torch.tensor(y)
            optim.zero_grad()
            yprime, h=  model.forward_train(input_vec)
            loss = lossFct(yprime.float(), output_vec.long())
            total_loss += loss.item()
            loss.backward()
            optim.step()
            n += 1
        print("\nAvg loss = {:.5f}".format(total_loss / n), file=sys.stderr)

def saveVocab(dicoVocab, fileName):
    f = open(fileName, "w")
    for form in dicoVocab :
        print(dicoVocab[form], form, file=f)

def main():
    if len(sys.argv) < 7:
        print("Usage :", sys.argv[0], "conlluFile modelName vocabFile BigramFile nbit embedding_dim")
        sys.exit(1)

    hidden_size = 256
    embedding_dim = int(sys.argv[6])
    nb_iter = int(sys.argv[5])

    conlluFileName = sys.argv[1]
    modelFileName = sys.argv[2]
    vocabFileName = sys.argv[3]
    bigramFile = sys.argv[4]

    dicoUpos = conlluPosTags()
    output_dim = len(dicoUpos)

    dicoVocab = extractVocab(conlluFileName)
    vocabSize = len(dicoVocab)

    LX, LY = prepareData(conlluFileName, dicoVocab, dicoUpos)

    model = GRUNet(embedding_dim, hidden_size, output_dim, vocabSize, bigramFile)

    train(LX, LY, model, nb_iter)
    torch.save(model, modelFileName)
    saveVocab(dicoVocab, vocabFileName)

if __name__ == '__main__':
    main()
