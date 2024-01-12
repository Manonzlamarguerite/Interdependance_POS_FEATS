#!/usr/bin/env bash
FILE_TEST="../../data/fr_gsd-ud-test.conllu"
FILE_TRAIN="../../data/fr_gsd-ud-train.conllu"
NAME_MODEL="train"
# NAME_PRED=""
T=10
NB_IT=1
DIM_EMB=100

# # Ajoute des mots inconnues dans le fichier d'entrainement
python3 ../../Outils/conllu_add_unk.py $FILE_TRAIN $T > ./input/$NAME_MODEL"."$T".conllu"

#---------- FEATS ---------- #
python3 ../../Model_indep_FEATS/Viterbi/conllu_bigrams.py ./input/$NAME_MODEL"."$T".conllu" "./big/"$NAME_MODEL"."$T".morpho.big"
echo "Début de l'entrainement FEATS"
# Entraine le modèle de prédiction de POS
python3 ../../Model_indep_FEATS/Viterbi/train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T".morpho.pt" "./voc/"$NAME_MODEL"."$T".morpho.voc" ./big/$NAME_MODEL"."$T".morpho.big" $NB_IT $DIM_EMB
echo "Fin de l'entraintement FEATS"

#---------- POS à partir de FEATS ---------- #
# Calcule les probabilité des bigrammes de POS
python3 conllu_bigrams.py ./input/$NAME_MODEL"."$T".conllu" "./big/$NAME_MODEL.$T.pos.big"
echo "Début de l'entrainement POS"
# # Entraine le modèle de prédiction de POS
python3 train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T".pos.pt" "./voc/"$NAME_MODEL"."$T".pos.voc" ./big/$NAME_MODEL"."$T".pos.big" $NB_IT $DIM_EMB
echo "Fin de l'entraintement POS"


# #---------- Inférence FEATS à partir des POS ---------- #
echo "Inférence des POS"
# Permet l'inférence sur le fichier de test des POS
python3 ../../Model_indep_FEATS/Viterbi/decode.py ./model/$NAME_MODEL"."$T".morpho.pt" $FILE_TEST "./voc/"$NAME_MODEL"."$T".morpho.voc" ./input/$NAME_MODEL"."$T".conllu.tagSet" > "./output/"$NAME_MODEL"."$T".auto.morpho.conllu"

echo "Inférence des FEATS"
# Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T".pos.pt" "./output/"$NAME_MODEL"."$T".auto.morpho.conllu" "./voc/"$NAME_MODEL"."$T".pos.voc" > "./output/"$NAME_MODEL"."$T".auto.pos.conllu"

echo "Evaluation du modèle"
# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.morpho.conllu" > "../../Evaluation/Viterbi/Model_Pipeline_FeatToPos/evaluation_fr_gsd-ud-test.conllu"

echo "Inférence des FEATS"
# Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T".pos.pt" $FILE_TEST "./voc/"$NAME_MODEL"."$T".pos.voc" ./input/$NAME_MODEL"."$T".conllu.tagSet" > "./output/"$NAME_MODEL"."$T".auto.morpho.conllu"

echo "Evaluation du modèle fausse (on donne les vrai POS)"
# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.morpho.conllu" > "../../Evaluation/Viterbi/Model_Pipeline_FeatToPos/evaluation_fausse_fr_gsd-ud-test.conllu"
