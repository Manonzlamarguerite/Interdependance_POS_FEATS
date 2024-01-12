import sys
from UDTagSet import UDTagSet
from bigrams import Bigrams

def main():
    if len(sys.argv) < 3:
        print("Usage :", sys.argv[0], "conlluFile bigramFile")
        sys.exit(1)

    conlluFileName  = sys.argv[1]    
    bigramFileName = sys.argv[2]

    tagSet = UDTagSet()
    bigrams = Bigrams()
    bigrams.computeFromConlluFile(conlluFileName, tagSet)
    bigrams.save(bigramFileName)

if __name__ == '__main__':
    main()
