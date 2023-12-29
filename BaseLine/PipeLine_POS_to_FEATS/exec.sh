#!/usr/bin/env bash

# # ---------- Prétraitement des données ---------- #
# # Ajoute des mots inconnues dans le fichier d'entrainement
# python3 conllu_add_unk.py fr_gsd-ud-train.conllu 10 > train.10.conllu
#
# # ---------- POS Tagging ---------- #
# # Calcule les probabilité des bigrammes de POS
# python3 ../Model_indep_POS/conllu_bigrams.py train.10.conllu train.10.big
#
# echo "Début de l'entrainement"
# # Entraine le modèle de prédiction de POS
# python3 ../Model_indep_POS/train.py train.10.conllu train.10.pt train.10.voc
# echo "Fin de l'entraintement"
#


# ---------- Traits morphologique à partir des POS ---------- #
# Calcule les probabilité des bigrammes de POS
python3 ../Model_indep_Morpho/conllu_bigrams.py train.10.conllu train.10.morpho.big

echo "Début de l'entrainement"
# Entraine le modèle de prédiction de POS
python3 train.py train.10.conllu train.10.morpho.pt train.10.morpho.voc
echo "Fin de l'entraintement"


# ---------- Inférence traits morphologique à partir des POS ---------- #


echo "Inférence des POS"
# Permet l'inférence sur le fichier de test des POS
python3 ../Model_indep_POS/decode.py train.10.pt fr_gsd-ud-test.conllu train.10.voc > test.10.auto.conllu

echo "Inférence des traits morphologique"
# Permet l'inférence sur le fichier de test
python3 decode.py train.10.morpho.pt test.10.auto.conllu train.10.morpho.voc train.10.conllu.tagSet > test.10.auto.morpho.conllu

echo "Evaluation du modèle"
# Permet d'évaluer le modèle
python3 conllu_eval.py fr_gsd-ud-test.conllu test.10.auto.morpho.conllu
