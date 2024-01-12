import sys
import os
from conllu import parse
from conllu import parse_incr

class UDTagSet:
    """
    - Paramètre :
    nameFile : nom du fichier d'entrainement où l'ont va récupérer les traits morphologiques
    tagSet : True si on veut crée le dico à partir du fichier
           False si on veut juste lire le fichier pour crée le dico
    """
    def __init__(self,nameFile, tagSet=False):
        self.dico = {}
        if tagSet:
            with open(nameFile, 'r') as fichier:
                for ligne in fichier:
                    line_seg = ligne.split()
                    self.dico[line_seg[0]] = int(line_seg[1])
        else:
            if os.path.exists(nameFile+".tagSet"):
                init = False
            else:
                init = True

            if init :
                data_file = open(nameFile, "r", encoding="utf-8")

                data_file = open(nameFile, "r", encoding="utf-8")
                self.dico['None'] = 0
                indice = 1
                for sentence in parse_incr(data_file):
                    for token in sentence :
                        feats = token['feats']

                        if feats != None:
                            code = self.dico2SuperTag(feats)
                            if not(code in self.dico.keys()):
                                self.dico[code] = indice
                                indice += 1

                self.writeFileTagSet(nameFile+".tagSet")

            else:
                with open(nameFile+".tagSet", 'r') as fichier:
                    for ligne in fichier:
                        line_seg = ligne.split()
                        self.dico[line_seg[0]] = int(line_seg[1])

    def writeFileTagSet(self, nameFileTagSet):
        with open(nameFileTagSet, "w") as file:
            for feats in self.dico.keys():
                file.write(feats+" "+str(self.dico[feats])+"\n")

    def dico2SuperTag(self, feats):
        superTag = ""

        if feats != None:
            for feat in feats.keys():
                superTag += ""+feat+"="+feats[feat]+"|"
            return superTag.rstrip('|')
        else:
            return 'None'

    def superTag2Dico(self, feats):
        if feats == 'None':
            return {}
        else:
            dico_f = {}
            feats_L = feats.split("|")
            for feat in feats_L:
                feat_L = feat.split("=")
                dico_f[feat_L[0]] = feat_L[1]
            return dico_f

    def codeToTag(self, indice):
        superTag = list(self.dico.keys())[indice]
        return self.superTag2Dico(superTag)

    def tagToCode(self, tag):
        supTag = self.dico2SuperTag(tag)
        if supTag in self.dico.keys():
            return self.dico[supTag]
        else:
            return 0

    def size(self):
        return len(self.dico.keys())

    def tags(self):
        return list(self.dico.keys())

def main():
    if len(sys.argv) < 2:
        print("Usage :", sys.argv[0], "conlluFile")
        sys.exit(1)

    conlluFileName = sys.argv[1]

    test = UDTagSet(conlluFileName)
    # test.testCode(conlluFileName)
    # print(test.dico)
    # print(test.size())

if __name__ == '__main__':
    main()
