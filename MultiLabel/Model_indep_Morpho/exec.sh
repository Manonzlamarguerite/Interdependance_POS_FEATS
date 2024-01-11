#!/usr/bin/env bash

# Ajoute des mots inconnues dans le fichier d'entrainement
# python3 conllu_add_unk.py fr_gsd-ud-train.conllu 10 > train.10.conllu

#echo "Début de l'entrainement"
# Entraine le modèle de prédiction de POS
#python3 train.py train.10.conllu train.10.pt train.10.voc
#echo "Fin de l'entraintement"
#
# # Permet l'inférence sur le fichier de test
#python3 decode.py train.10.pt fr_gsd-ud-test.conllu train.10.voc train.10.conllu.tagSet  > train.10.auto.conllu
#echo "Evaluation du modèle"
# Permet d'évaluer le modèle
python3 conllu_eval.py fr_gsd-ud-test.conllu train.10.auto.conllu
