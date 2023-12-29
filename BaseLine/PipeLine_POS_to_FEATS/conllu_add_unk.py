import sys
from conllu import parse
from conllu import parse_incr



def replaceRareWords(conlluFileName, threshold) :

    counts = {}
    data_file = open(conlluFileName, "r", encoding="utf-8")
    wordsNb = 0

    for sentence in parse_incr(data_file):

        for token in sentence :
            form = token['form']
            wordsNb += 1
            if form not in counts:
                counts[form] = 1
            else :
                counts[form] += 1
    data_file.close()


    wordsReplaced = 0
    data_file = open(conlluFileName, "r", encoding="utf-8")

    for sentence in parse_incr(data_file):
        for token in sentence :
            form = token['form']
            if counts[form] < threshold:
                token['form'] = '<UNK>'
                wordsReplaced += 1
        print(sentence.serialize())
    data_file.close()
    return wordsNb, wordsReplaced

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage :", sys.argv[0], "conlluFile threshold")
        sys.exit(1)

    conlluFileName = sys.argv[1]
    threshold = int(sys.argv[2])
    wordsNb, wordsReplaced = replaceRareWords(conlluFileName, threshold)
    print(wordsReplaced, "occurrences have been replaced with <UNK>, over", wordsNb, "ratio = ", wordsReplaced/wordsNb, file=sys.stderr)
