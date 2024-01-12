#!/usr/bin/env bash
FILE_TEST="../../data/fr_gsd-ud-test.conllu"
FILE_TRAIN="../../data/fr_gsd-ud-train.conllu"
NAME_MODEL="train"

T=10
NB_IT=1
DIM_EMB=100
DIM_CHAR_EMB=100


# # Ajoute des mots inconnues dans le fichier d'entrainement
# python3 ../../Outils/conllu_add_unk.py $FILE_TRAIN $T > ./input/$NAME_MODEL"."$T".conllu"
#
# #---------- POS ---------- #
# echo "Début de l'entrainement POS"
# # Entraine le modèle de prédiction des traits morphologiques
# python3 ../../Model_indep_POS/BILSTMConvolution/train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T".pos.pt" "./voc/"$NAME_MODEL"."$T".pos.voc" $NB_IT $DIM_EMB $DIM_CHAR_EMB "./voc/"$NAME_MODEL"char."$T".pos.voc"
# echo "Fin de l'entraintement POS"
#
# #----------FEATS à partir de POS ---------- #
# echo "Début de l'entrainemen FEATS"
# # Entraine le modèle de prédiction de POS
# python3 train.py ./input/$NAME_MODEL"."$T".conllu" ./model/$NAME_MODEL"."$T".morpho.pt" "./voc/"$NAME_MODEL"."$T".morpho.voc" $NB_IT $DIM_EMB $DIM_CHAR_EMB "./voc/"$NAME_MODEL"char."$T".morpho.voc"
# echo "Fin de l'entraintement  FEATS"
#
# # ############
# echo "Inférence des traits morphologiques"
# # Permet l'inférence sur le fichier de test des POS
python3 ../../Model_indep_POS/BILSTMConvolution/decode.py ./model/$NAME_MODEL"."$T".pos.pt" $FILE_TEST ./voc/$NAME_MODEL"."$T".pos.voc"  ./voc/$NAME_MODEL"char."$T".pos.voc" > "./output/"$NAME_MODEL"."$T".auto.pos.conllu"
#
# echo "Inférence des POS"
# # Permet l'inférence sur le fichier de test
python3 decode.py ./model/$NAME_MODEL"."$T".morpho.pt" ./output/$NAME_MODEL"."$T".auto.pos.conllu" ./voc/$NAME_MODEL"."$T".morpho.voc" ./input/$NAME_MODEL"."$T".conllu.tagSet"  ./voc/$NAME_MODEL"char."$T".morpho.voc" > "./output/"$NAME_MODEL"."$T".auto.morpho.conllu"
#
# # Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.morpho.conllu" > "../../Evaluation/BiLSTM_Conv/Model_Pipeline_PosToFeat/evaluation_fr_gsd-ud-test.conllu"
#
python3 decode.py ./model/$NAME_MODEL"."$T".morpho.pt" $FILE_TEST ./voc/$NAME_MODEL"."$T".morpho.voc" ./input/$NAME_MODEL"."$T".conllu.tagSet"  ./voc/$NAME_MODEL"char."$T".morpho.voc" > "./output/"$NAME_MODEL"."$T".auto.morpho.faux.conllu"


# Permet d'évaluer le modèle
python3 ../../Outils/conll18_ud_eval.py -v $FILE_TEST "./output/"$NAME_MODEL"."$T".auto.morpho.faux.conllu" > "../../Evaluation/BiLSTM_Conv/Model_Pipeline_PosToFeat/evaluation_fausse_fr_gsd-ud-test.conllu"
