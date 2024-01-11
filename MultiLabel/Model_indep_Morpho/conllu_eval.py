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
	ok = 0
	total = 0
	for sentRef, sentHyp in zip(parse_incr(ref), parse_incr(hyp)):
		for tokenRef, tokenHyp in zip(sentRef, sentHyp):
			total += 1
			featsRef = tokenRef["feats"]
			featsHyp = tokenHyp["feats"]
			print("###")
			print(featsHyp, featsRef)
			if featsRef == featsHyp:
				print("OK")
				ok += 1
	accuracy = ok / total
	print("accuracy = ", accuracy)

if __name__ == '__main__':
    main()
