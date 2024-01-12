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
                indice = 0

                for sentence in parse_incr(data_file):
                    for token in sentence :
                        feats = token['feats']
                        if feats != None:
                            for feat in feats:
                                label = feat+"="+feats[feat]
                                if not(label in self.dico.keys()):
                                    self.dico[label] = indice
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

    def codeToTag(self, code):
        i = 0
        tag = ""
        for feat in self.dico.keys():
            if code[i] == 1:
                tag += feat+"|"
            i += 1
        return tag[:-1]

    def tagToCode(self, dico_tag):
        code = []
        if dico_tag != None:
            for feat_val in self.dico.keys():
                s = feat_val.split("=")
                feat = s[0]
                val = s[1]
                if feat in dico_tag.keys():
                    if dico_tag[feat] == val:
                        code.append(1)
                    else:
                        code.append(0)
                else:
                    code.append(0)
        else :
            for i in range(len(self.dico.keys())):
                code.append(0)
        return code

    def size(self):
        return len(self.dico.keys())

    def tags(self):
        return list(self.dico.keys())

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
    # data_file = open(conlluFileName, "r", encoding="utf-8")
    # c=0
    # for sentence in parse_incr(data_file):
    #     if c == 0 :
    #         for token in sentence :
    #             feats = token['feats']
    #             print(token, feats)
    #             print(test.tagToCode(feats))
    #     c += 1

    # print(test.dico)
    # print(test.size())

if __name__ == '__main__':
    main()
