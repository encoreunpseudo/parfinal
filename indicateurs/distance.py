import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
from scipy.stats import norm
import streamlit as st

# Dossiers contenant les CSV
dossier_csv_1 = "./csv_essai"  # Premier dossier
dossier_csv_2 = "./csv_essai"  # Deuxième dossier

# Choisir les classes pour l'analyse des distances
classe_1 = "OP"  # Classe 1
classe_2 = "mains"  # Classe 2

# Fonction pour extraire les coordonnées de la boîte
def extract_box_coordinates(box_str):
    return ast.literal_eval(box_str)

def analyser_distances(dossier_csv, titre):
    toutes_distances = []

    for fichier in os.listdir(dossier_csv):
        if fichier.endswith(".csv"):
            df = pd.read_csv(os.path.join(dossier_csv, fichier))

            df[['x_min', 'y_min', 'x_max', 'y_max']] = df['box'].apply(extract_box_coordinates).apply(pd.Series)

            df['center_x'] = (df['x_min'] + df['x_max']) / 2
            df['center_y'] = (df['y_min'] + df['y_max']) / 2

            df_classe_1 = df[df['class'] == classe_1]
            df_classe_2 = df[df['class'] == classe_2]

            for frame in df_classe_1['frame'].unique():
                objets_classe_1_frame = df_classe_1[df_classe_1['frame'] == frame]
                objets_classe_2_frame = df_classe_2[df_classe_2['frame'] == frame]

                for _, obj_1 in objets_classe_1_frame.iterrows():
                    for _, obj_2 in objets_classe_2_frame.iterrows():
                        distance = np.sqrt((obj_1['center_x'] - obj_2['center_x'])**2 + (obj_1['center_y'] - obj_2['center_y'])**2)
                        toutes_distances.append(distance)

    mean_distance = np.mean(toutes_distances)
    std_distance = np.std(toutes_distances)

    x = np.linspace(min(toutes_distances), max(toutes_distances), 100)
    y = norm.pdf(x, mean_distance, std_distance) * len(toutes_distances) * (max(toutes_distances) - min(toutes_distances)) / 30

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(toutes_distances, bins=30, color='b', edgecolor='black', alpha=0.7, density=False)
    ax.plot(x, y, 'r-', label=f"Courbe gaussienne ajustée\nMoyenne = {mean_distance:.2f}\nÉcart-type = {std_distance:.2f}")
    ax.legend(loc="upper right")
    ax.set_title(f"Distribution des distances entre {classe_1} et {classe_2} ({titre})")
    ax.set_xlabel("Distance (pixels)")
    ax.set_ylabel("Fréquence")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Affichage dans Streamlit
    st.pyplot(fig)

# Afficher le titre principal une seule fois
st.title("Analyse des distances entre OP et mains")

# Exécution pour les deux dossiers
analyser_distances(dossier_csv_1, "Alice 6 mois")
analyser_distances(dossier_csv_2, "Alice 12 mois")
