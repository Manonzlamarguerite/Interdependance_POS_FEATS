import sys
from conllu import parse
from conllu import parse_incr

def main():
	if len(sys.argv) < 3:
		print("Usage :", sys.argv[0], "ref_file hyp_file")
		sys.exit(1)

	conlluRefFileName = sys.argv[1]
	conlluHypFileName = sys.argv[2]
	ref = open(conlluRefFileName, "r", encoding="utf-8")
	hyp = open(conlluHypFileName, "r", encoding="utf-8")
	ok_feat = 0
	ok_pos = 0
	total = 0
	for sentRef, sentHyp in zip(parse_incr(ref), parse_incr(hyp)):
		for tokenRef, tokenHyp in zip(sentRef, sentHyp):
			total += 1
			featsRef = tokenRef["feats"]
			featsHyp = tokenHyp["feats"]

			posRef = tokenRef["upos"]
			posHyp = tokenHyp["upos"]

			if featsRef == featsHyp:
				ok_feat += 1
			if posHyp == posRef:
				ok_pos += 1

	accuracy_feat = ok_feat / total
	accuracy_pos = ok_pos / total
	print("accuracy (feats) = ", accuracy_feat, "accuracy (pos) =", accuracy_pos)

if __name__ == '__main__':
    main()
