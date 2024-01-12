import sys
from conllu import parse
from conllu import parse_incr
from torch.nn.utils.rnn import pad_sequence

from utils import conlluPosTags
import torch
import torch.nn as nn
from BILSTMConvolution import BiLSTMConvolutionTagger
from UDTagSet import UDTagSet



def extractVocab(conlluFileName):
    dicoVocab = {}
    data_file = open(conlluFileName, "r", encoding="utf-8")
    dicoVocab["<UNK>"] = 0
    vocabSize = 1
    for sentence in parse_incr(data_file):
        for token in sentence :
            form = token['form']
            form = form+"|"+token['upos']
            if form not in dicoVocab:
                dicoVocab[form] = vocabSize
                vocabSize += 1
    data_file.close()
    return dicoVocab

def extractCharVocab (ConlluFilename):
    # Load the CoNLL-U dataset
    with open(ConlluFilename, 'r', encoding='utf-8') as file:
        data = file.read()
    sentences = parse(data)
    vocab_character_list = {}
    vocab_character_list["<UNK>"] = 0
    i=1
    for sentence in sentences:
        for token in sentence:
             if isinstance(token, dict):
                word = token['form']
                word = word+"|"+token['upos']
                for char in word :
                    if char not in vocab_character_list.keys():
                        vocab_character_list[char]=i+1
                        i+=1

    return vocab_character_list

def prepareData(conlluFileName, dicoCharVocab,dicoVocab, tagSet) :
    data_file = open(conlluFileName, "r", encoding="utf-8")
    LX = []
    LY = []
    LC = []
    for sentence in parse_incr(data_file):
        X = []
        Y = []
        C = []
        for token in sentence :
            Char = []
            form = token['form']
            form = form+"|"+token['upos']
            if form not in dicoVocab :
                formCode = dicoVocab['<UNK>']
            else :
                formCode = dicoVocab[form]
            X.append(formCode)
            feats = token['feats']
            featsCode = tagSet.tagToCode(feats)
            Y.append(featsCode)
            for char in form:
                Char.append(dicoCharVocab[char])
            C.append(Char)
        LC.append(C)
        LX.append(X)
        LY.append(Y)
    return LX, LY,LC

def train(LX, LY,LC, model, nb_iter):
    optim = torch.optim.Adam(model.parameters())
    lossFct = torch.nn.CrossEntropyLoss()
    for iteration in range(nb_iter):
        total_loss = 0
        print("iteration : ", iteration + 1, file=sys.stderr)
        n = 0
        for (x,y,c) in zip(LX,LY,LC):
            if n % 100 == 0 :
                print(".", end="", flush=True, file=sys.stderr)
            input_vec = torch.tensor(x)

            input_char = pad_sequence([torch.tensor(seq) for seq in c], batch_first=True, padding_value=0)

            output_vec = torch.tensor(y)
            optim.zero_grad()
            yprime, _=  model.forward(input_vec, input_char)

            loss = lossFct(yprime, output_vec)
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
    if len(sys.argv) < 8:
        print("Usage :", sys.argv[0], "conlluFile modelName vocabFile nb_it embedding_dim vocabChar")
        sys.exit(1)

    hidden_size = 256
    embedding_dim = int(sys.argv[5])
    CharEmbdDim = int(sys.argv[6])
    nb_iter = int(sys.argv[4])

    conlluFileName = sys.argv[1]
    modelFileName = sys.argv[2]
    vocabFileName = sys.argv[3]
    vocabCharFileName = sys.argv[7]

    tagSet = UDTagSet(conlluFileName)
    output_dim = tagSet.size()

    dicoVocab = extractVocab(conlluFileName)
    vocabSize = len(dicoVocab)
    dicoCharVocab = extractCharVocab(conlluFileName)
    vocabCharSize = len(dicoCharVocab) + 1
    LX, LY, LC = prepareData(conlluFileName, dicoCharVocab,dicoVocab, tagSet)

    model = BiLSTMConvolutionTagger(embedding_dim, CharEmbdDim,hidden_size, output_dim, vocabSize,vocabCharSize)

    train(LX, LY,LC,model, nb_iter)
    torch.save(model, modelFileName)
    saveVocab(dicoVocab, vocabFileName)
    saveVocab(dicoCharVocab, vocabCharFileName)

if __name__ == '__main__':
    main()
