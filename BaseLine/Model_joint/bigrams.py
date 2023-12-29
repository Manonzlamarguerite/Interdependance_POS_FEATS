import sys
from conllu import parse
from conllu import parse_incr
import numpy as np
from UDTagSet import UDTagSet

class Bigrams():
    def __init__(self):
        self.bigramMatrix = None
        self.initialProb = None
        self.unigramVector = None
        self.tagSet = None

    def addTagSet(self, tagSet):
        self.tagSet = tagSet

    def getTagSet(self) :
        return self.tagSet

    def computeFromConlluFile(self, conlluFileName, tagSet):
    #def computeBigramMatrix(conlluFileName, tagSet) :
        self.bigramMatrix = np.zeros((tagSet.size(),tagSet.size()))
        unigramVector = np.zeros(tagSet.size())
        self.initialProb = np.zeros(tagSet.size())
        self.tagSet = tagSet

        data_file = open(conlluFileName, "r", encoding="utf-8")
        for sentNb, sentence in enumerate(parse_incr(data_file)):
            precFeats = None
            for token in sentence :
                currFeats = token['feats']
                if currFeats != None:
                    currFeats["POS"] = token['upos']
                else:
                    currFeats = {}
                    currFeats["POS"] = token['upos']
                currFeatsCode = tagSet.tagToCode(currFeats)

                if not precFeats :
                    self.initialProb[currFeatsCode] += 1
                else :
                    self.bigramMatrix[precFeatsCode][currFeatsCode] += 1

                unigramVector[currFeatsCode] += 1
                precFeats = currFeats
                precFeatsCode = currFeatsCode

        for precFeats in range(tagSet.size()):
            for currFeats in range(tagSet.size()):
                self.bigramMatrix[precFeats][currFeats] /= unigramVector[precFeats]
        for currFeatsCode in range(tagSet.size()):
            self.initialProb[currFeatsCode] /= sentNb


    def bigramProba(self, bigramMatrix, precFeats, currFeats):
        return self.bigramMatrix[self.tagSet.tagToCode(precFeats)][self.tagSet.tagToCode(currFeats)]


    def save(self, fileName):
        f = open(fileName, "w")
        tagSetSize = self.tagSet.size()
        print("#VOCAB", tagSetSize, file=f)
        for index, Feats in enumerate(self.tagSet.tags()):
            print(index, Feats, file=f)
        print("#INITIAL", file=f)
        for Feats in range(tagSetSize):
            print("{:.4f}".format(self.initialProb[Feats]), file=f)
        print("#BIGRAM", file=f)
        for precFeats in range(tagSetSize):
            for currFeats in range(tagSetSize):
                print("{:.4f}".format(self.bigramMatrix[precFeats][currFeats]), file=f)
        print("#END", file=f)


    def load(self, fileName):
        f = open(fileName, "r")
        line = f.readline()
        if "VOCAB" in line :
            vocab = {}
            line = f.readline()
            line.rstrip()
        while line[0] != "#":
            code, symbol = line.split()
            vocab[symbol] = int(code)
            line = f.readline()
            line.rstrip()
        if "INITIAL" in line :
            self.initialProb = np.zeros(len(vocab))
            line = f.readline()
            line.rstrip()
            i = 0
            while line[0] != "#":
                self.initialProb[i] = float(line)
                i += 1
                line = f.readline()
                line.rstrip()
        if "BIGRAM" in line :
            probas = []
            line = f.readline()
            line.rstrip()
        while line[0] != "#":
            probas.append(float(line))
            line = f.readline()
            line.rstrip()
        self.bigramMatrix = np.array(probas).reshape(len(vocab), len(vocab))
        f.close()
