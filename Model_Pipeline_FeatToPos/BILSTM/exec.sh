#!/usr/bin/env bash
FILE_TEST="../../data/fr_gsd-ud-test.conllu"
FILE_TRAIN="../../data/fr_gsd-ud-train.conllu"
NAME_MODEL="train"
# NAME_PRED=""
T=10
NB_IT=5
DIM_EMB=100

# # Ajoute des mots inconnues dans le fichier d'entrainement
python3 ../../Outils/conllu_add_unk.py $FILE_TRAIN $T > ./input/$NAME_MODEL"."$T".conllu"
#
# ---------- Trais morphologique ---------- #
echo "Début de l'entrainement FEATS"
# Entraine le modèle de prédiction des traits morphologiques
python3 ../../Model_indep_FEATS/BILSTM/train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T"morpho.pt" "./voc/"$NAME_MODEL"."$T"morpho.voc" $NB_IT $DIM_EMB
echo "Fin de l'entraintement FEATS"


#----------POS à partir des trais morpho ---------- #
echo "Début de l'entrainement POS"
# Entraine le modèle de prédiction de POS
python3 train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T"pos.pt" "./voc/"$NAME_MODEL"."$T"pos.voc" $NB_IT $DIM_EMB
echo "Fin de l'entraintement POS"

#---------- InférencePOS à partr des traits morphologiques---------- #
echo "Inférence des traits morphologiques"
# Permet l'inférence sur le fichier de test des POS
python3 ../../Model_indep_FEATS/BILSTM/decode.py ./model/$NAME_MODEL"."$T"morpho.pt" $FILE_TEST "./voc/"$NAME_MODEL"."$T"morpho.voc"  ./input/$NAME_MODEL"."$T".conllu.tagSet" > "./output/"$NAME_MODEL"."$T".auto.morpho.conllu"

echo "Inférence des POS"
# Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T"pos.pt" "./output/"$NAME_MODEL"."$T".auto.morpho.conllu" "./voc/"$NAME_MODEL"."$T"pos.voc" > "./output/"$NAME_MODEL"."$T".auto.pos.conllu"

echo "Evaluation du modèle"
# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.pos.conllu" > "../../Evaluation/BiLSTM/Model_Pipeline_FeatToPos/evaluation_fr_gsd-ud-test.conllu"

echo "Evaluation du modèle fausse (on donne les vrais POS)"
# Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T"pos.pt" $FILE_TEST "./voc/"$NAME_MODEL"."$T"pos.voc" > "./output/"$NAME_MODEL"."$T".auto.faux.pos.conllu"

# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.faux.pos.conllu" > "../../Evaluation/BiLSTM/Model_Pipeline_FeatToPos/evaluation_fausse_fr_gsd-ud-test.conllu"
