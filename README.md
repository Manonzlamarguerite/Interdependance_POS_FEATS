# Interdependance_POS_FEATS
Ce projet a pour but d'étudier les interdépendances entre les tâches de POS Tagging et la prédiction de traits morphologiques (_feats_).

Nous souhaitons étudier les indépendances possibles entre les tâches de prédiction des étiquettes POS et de prédiction des traits morphologiques. Pour ce faire, nous avons créé 4 types de modèles :
- **Modèle indépendant pour les POS** : un modèle qui permet de prédire uniquement les étiquettes POS.
- **Modèle indépendant pour les FEATS** : un modèle qui permet de prédire uniquement les traits morphologiques.
- **Modèle pipeline de POS vers FEATS** : un modèle en pipeline où l'objectif est d'utiliser les étiquettes POS comme information pour le modèle de prédiction des traits morphologiques.
- **Modèle pipeline de FEATS vers POS** : un modèle en pipeline où l'objectif est d'utiliser les traits morphologiques comme information pour le modèle de prédiction des étiquettes POS.
- **Modèle joint** : un modèle qui cherche à prédire les étiquettes POS et les traits morphologiques simultanément.

Pour la prédiction des traits morphologiques, nous avons décidé d'utiliser une classe par supertag présent dans les données d'entraînement.

# Architectures utilisées

Nous avons décidé de tester différentes architectures pour chaque type de modèle afin d'avoir plusieurs visions différentes pour l'étude des dépendances entre nos tâches. Ainsi, pour chaque type de tâche, nous avons implémenté chacune des architectures suivantes :
- **Baseline** : qui est un modèle de type GRU.
- **Viterbi** : qui est un modèle de type GRU, auquel on ajoute l'algorithme de Viterbi lors de l'apprentissage.
- **MultiLab** : qui est un modèle de type GRU, mais où nous avons choisi de faire de la classification multi-label à la place d'utiliser une classe par supertag pour les traits morphologiques. À noter que cette architecture n'existe pas pour le type indépendant pour les POS.
- **BiLSTM** : qui est un modèle de type BiLSTM.
- **BiLSTM_Conv** : qui est un modèle de type BiLSTM avec une couche d'embedding sur les caractères.

# Structure du projet

Voici la liste des dossiers que vous trouverez :
- `data` : il contient les fichiers UD d'entraînement et de test utilisés.
- `Evaluation` : ce dossier est lui-même divisé en plusieurs dossiers. Chaque dossier correspond à une architecture de modèle différente. Au sein de chaque dossier portant sur une architecture, vous retrouverez un dossier par type de modèle.
- `Outils` : des scripts Python pour l'évaluation et l'ajout de mots inconnus.
- `Model_indep_POS` : où l'on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type indépendant pour les POS.
- `Model_indep_FEATS` : où l'on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type indépendant pour les FEATS.
- `Model_Pipeline_PosToFeat` : où l'on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type pipeline POS vers FEATS.
- `Model_Pipeline_FeatToPos` : où l'on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type pipeline FEATS vers POS.
- `Model_joint` : où l'on trouve les différentes architectures que nous avons décidé d'implémenter et permettant la création d'un modèle de type joint.
