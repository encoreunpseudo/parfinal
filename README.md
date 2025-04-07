# Système de Détection et d'Analyse Vidéo pour l'Étude du Développement Cognitif

Ce projet propose un pipeline complet pour la détection, la classification, l'analyse et la visualisation d'objets présents dans des vidéos d'enfants en interaction. Il repose sur une architecture modulaire basée sur le deep learning (YOLOv8 fine-tuné) et inclut des métriques spatiales et temporelles utiles à l'étude du développement cognitif.

## Fonctionnalités principales

- **Détection automatique** de 6 classes d'objets dans les vidéos
- **Analyse spatiale et temporelle** des interactions
- **Visualisation interactive** des résultats via Streamlit
- **Exportation des données** sous forme de tableaux CSV

## Structure du Projet

```
├── detector/
│   ├── classes.csv     # Définitions des classes et assignation YOLO
│   ├── denoise          # Débruitage vidéo
│   ├── main2.py             # Pipeline principal de traitement
│   ├── run.py               # Script de lancement
│   ├── yolov8_model_new.pt  # Modèle YOLOv8 fine-tuné pour la détection des OP
├── exemples/            # Résultats visuels des vidéos traitées
│   ├── output.mp4     # Exemple de vidéo
├── indicateurs/
│   ├── distance.py          # Calcul de la distance entre mains et objets petits (OP)
│   ├── pos_moyenne.py       # Moyennes spatiales des classes détectées
│   └── tps_presence.py      # Temps moyen de présence par classe
```

## Protocole de Traitement Vidéo

L'analyse vidéo suit un protocole en **quatre étapes successives** :

### 1. Prétraitement
- Débruitage du signal visuel pour améliorer la qualité de détection
- Préparation des frames pour l'analyse

### 2. Détection d'Objets
- Utilisation d'un modèle **YOLOv8 fine-tuné** sur nos données spécifiques
- Détection des **6 classes opérationnelles** :
  - `visages` : visages des participants
  - `mains` : mains des participants
  - `A` : animaux
  - `OAG` : objets artificiels grands
  - `ONG` : objets naturels grands
  - `OP` : objets petits (fusion naturels / artificiels)

### 3. Structuration des Données
- Résultats organisés sous forme de **tableaux CSV** incluant:
  - Classes détectées
  - Coordonnées spatiales
  - Timestamps
  - Scores de confiance

### 4. Visualisation et Analyse
- Interfaces interactives pour explorer les données
- Métriques avancées sur les comportements et interactions

## Indicateurs fournis

| Script | Fonction | Application |
|--------|----------|-------------|
| `distance.py` | Mesure la distance entre les mains et les objets petits | Analyse des interactions enfant-objet |
| `pos_moyenne.py` | Calcule la position moyenne des classes dans l'espace | Cartographie de l'espace d'interaction |
| `tps_presence.py` | Estime le temps moyen de présence de chaque classe | Analyse de l'attention et de l'engagement |

## Guide d'utilisation

### 1. Lancer le traitement principal
```bash
python detector/run.py
```

### 2. Exécuter les analyses via Streamlit
```bash
# Calcul des distances
streamlit run indicateurs/distance.py

# Position moyenne
streamlit run indicateurs/pos_moyenne.py

# Temps de présence
streamlit run indicateurs/tps_presence.py
```

## Données d'entrée et de sortie

- **Entrée**: Les vidéos à analyser doivent être placées dans le dossier `exemples/` au format `.mp4`
- **Sortie**: 
  - Frames annotées (disponibles dans `exemples/`)
  - Fichiers CSV des métriques
  - Visualisations interactives via Streamlit

## Contexte de recherche

Ce pipeline a été développé dans le cadre d'un projet de recherche à l'École Centrale de Lyon sur le développement cognitif. Il vise à automatiser l'étude des interactions enfant-objet et à produire des indicateurs exploitables pour les chercheurs en sciences cognitives.
