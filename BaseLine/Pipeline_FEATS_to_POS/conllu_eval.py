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
	ok_pos = 0
	ok_feat = 0
	total = 0
	for sentRef, sentHyp in zip(parse_incr(ref), parse_incr(hyp)):
		for tokenRef, tokenHyp in zip(sentRef, sentHyp):
			total += 1
			featsRef = tokenRef["feats"]
			featsHyp = tokenHyp["feats"]
			posRef = tokenRef["upos"]
			posHyp = tokenHyp["upos"]

			# print("FEATS",featsRef,featsHyp)
			# print("POS", posRef, posHyp)
			if posRef == posHyp:
				ok_pos += 1
			if featsHyp == featsRef:
				ok_feat += 1

	accuracy_pos = ok_pos / total
	accuracy_feat = ok_feat / total
	print("accuracy POS = ", accuracy_pos, "(utilisé)accuracy FEATS = ", accuracy_feat)

if __name__ == '__main__':
    main()
