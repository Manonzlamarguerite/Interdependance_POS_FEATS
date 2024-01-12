#!/usr/bin/env bash
FILE_TEST="../../data/fr_gsd-ud-test.conllu"
FILE_TRAIN="../../data/fr_gsd-ud-train.conllu"
NAME_MODEL="train"
# NAME_PRED=""
T=10
NB_IT=1
DIM_EMB=100

# Ajoute des mots inconnues dans le fichier d'entrainement
python3 ../../Outils/conllu_add_unk.py $FILE_TRAIN $T > ./input/$NAME_MODEL"."$T".conllu"

echo "Début de l'entrainement"
# Entraine le modèle de prédiction de POS
python3 train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T".pt" "./voc/"$NAME_MODEL"."$T".voc" $NB_IT $DIM_EMB
echo "Fin de l'entraintement"

echo "Evaluation du modèle"
# Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T".pt" $FILE_TEST ./voc/$NAME_MODEL"."$T".voc" ./input/$NAME_MODEL"."$T".conllu.tagSet" > "./output/"$NAME_MODEL"."$T".auto.conllu"
# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.conllu"> "../../Evaluation/Baseline/Model_indep_FEATS/evaluation_fr_gsd-ud-test.conllu"
