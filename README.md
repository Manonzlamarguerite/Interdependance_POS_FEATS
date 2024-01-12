# Prediction_traits_morphologiques
Ce projet a pour but d'étudier l'interdépendances entres les tâches de POS Tagging et prédiction de traits morphologiques.

Nous souhaitons étudier les indépendances possibles entre les tâches de prédiction des étiquettes POS, et de prédiction des traits morphologiques. Pour ce faire, nous avons fait 4 types de modèles :
- **Modèle indépendant pour les POS** : un modèle qui permet de prédire uniquement les étiquettes POS.
- **Modèle indépendant pour les FEATS** : un modèle qui permet de prédire uniquement les traits morphologiques.
- **Modèle pipeline de POS vers FEATS** : un modèle en pipeline où l'objectif est d'utiliser les étiquettes POS comme information pour le modèle de prédiction des traits morphologiques.
- **Modèle pipeline de FEATS vers POS** : un modèle en pipeline où l'objectif est d'utiliser les traits morphologiques comme information pour le modèle de prédiction des étiquettes POS.
- **Modèle joint** : un modèle qui cherche à prédire les étiquettes POS et les traits morphologiques en simultané.

Pour la prédiction des traits morphologiques nous avons décidé d'utiliser une classe par supertag présent dans les données d'entrainement.

# Architectures utilisées

Nous avons décidé de tester différentes architectures pour chaque type de modèle afin d'avoir plusieurs visions différentes pour l'études des dépendances entre nos tâches. Ainsi, pour chaque type de tâche nous avons implémenter chacune des architectures suivantes :
- **Baseline** : qui est un modèle type GRU.
- **Viterbi** : qui est un modèle type GRU, auquel on ajoute l'algorithme de Viterbi lors de l'apprentissage.
- **MultiLab** : qui est un modèle type GRU, mais où nous avons choisi de fait de la classification multi-label à la place d'utiliser une classe par supertag pour les traits morphologiques. A noter que cette achitecture n'existe pas pour le type indépendant pour les POS.
- **BiLSTM** : qui est un modèle type BiLSTM.
- **BiLSTM_Conv** : qui est un modèle type BiLSTM avec un couche d'embedding sur les caractères.

# Structure du projet

Voici la liste des dossiers que vous trouverez :
- `data` : il contient les fichiers UD d'entrainement et de test utilisés.
- `Evaluation` : ce dossier est lui-même divisé en plusieurs dossiers. Chaque dossier correspond à une architecture de modèle différentes. Au sein de chaque dossier portant sur architecture, vous retrouverez un dossier par type de modèle.
- `Outils` : des scripts python pour l'évaluation et l'ajout de mots inconnus.
- `Model_indep_POS` : où on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type indépendant pour les POS.
- `Model_indep_FEATS` : où on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type indépendant pour les FEATS.
- `Model_Pipeline_PosToFeat` : où on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type pipeline POS vers FEATS.
- `Model_Pipeline_FeatToPos` : où on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type pipeline FEATS vers POS.
- `Model_joint` : où on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type joint.
