#!/bin/bash

# Définir le chemin du dossier à analyser
projet_dossier="."

# Lister tous les sous-dossiers
echo "Liste des dossiers dans $projet_dossier:"
for type_modele in $projet_dossier/*/
do
    # Supprimer le chemin et le slash final pour obtenir uniquement le nom du dossier
    nom_dossier=$(basename "$type_modele")

    # Vérifier si le nom du dossier est "test" ou "test2"
    if [[ $nom_dossier != "Evaluation" && $nom_dossier != "data" && $nom_dossier != "Outils" ]]; then
      echo "Modèle : ${nom_dossier}"
      for type_architecture in $type_modele*
      do
          cd $type_architecture
          nom_architecture=$(basename "$type_architecture")
          echo "Architecture : ${nom_architecture}"
          time bash exec.sh 
          echo ""
          cd ../..
      done
    fi
done
