#!/usr/bin/env bash

# ---------- Prétraitement des données ---------- #
# Ajoute des mots inconnues dans le fichier d'entrainement
python3 conllu_add_unk.py fr_gsd-ud-train.conllu 10 > train.10.conllu

# ---------- Trais morphologique ---------- #
# echo "Début de l'entrainement"
# # Entraine le modèle de prédiction des traits morphologiques
# python3 ../Model_indep_Morpho/train.py train.10.conllu train.10.pt train.10.voc
# echo "Fin de l'entraintement"

# ----------POS à partir des trais morpho ---------- #
echo "Début de l'entrainement"
# Entraine le modèle de prédiction de POS
python3 train.py train.10.conllu train.10.pos.pt train.10.pos.voc
echo "Fin de l'entraintement"


# ---------- InférencePOS à partr des traits morphologiques---------- #
echo "Inférence des traits morphologiques"
# Permet l'inférence sur le fichier de test des POS
python3 ../Model_indep_Morpho/decode.py ../Model_indep_Morpho/train.10.pt fr_gsd-ud-test.conllu ../Model_indep_Morpho/train.10.voc ../Model_indep_Morpho/train.10.conllu.tagSet > test.10.auto.conllu

echo "Inférence des POS"
# Permet l'inférence sur le fichier de test
python3 decode.py train.10.pos.pt test.10.auto.conllu train.10.pos.voc > test.10.auto.pos.conllu

echo "Evaluation du modèle"
# Permet d'évaluer le modèle
python3 conllu_eval.py fr_gsd-ud-test.conllu test.10.auto.pos.conllu
