import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import os
import ast
import cv2

# Définition des dossiers CSV
dossier_csv_1 = "./csv_essai"
dossier_csv_2 = "./csv_essai"

# Classes à analyser
classes_recherchees = ["mains", "visages", "P", "OP", "ONP", "ONG", "OAG"]

# Définir des couleurs distinctes pour chaque classe
couleurs_classes = {
    "mains": (0.0, 0.0, 1.0),  # Bleu
    "visages": (1.0, 0.0, 0.0),  # Rouge
    "P": (0.0, 1.0, 0.0),  # Vert
    "OP": (1.0, 1.0, 0.0),  # Jaune
    "ONP": (1.0, 0.0, 1.0),  # Magenta
    "ONG": (0.0, 1.0, 1.0),  # Cyan
    "OAG": (0.5, 0.5, 0.0)   # Olive
}

# Fonction pour extraire les coordonnées des boxes
def extract_box_coordinates(box_str):
    return ast.literal_eval(box_str)

# Fonction pour analyser les positions moyennes et variances pour chaque classe
def analyser_position_moyenne(dossier_csv):
    positions_moyennes_x = {classe: [] for classe in classes_recherchees}
    positions_moyennes_y = {classe: [] for classe in classes_recherchees}
    variances_x = {classe: [] for classe in classes_recherchees}
    variances_y = {classe: [] for classe in classes_recherchees}
    
    for fichier in os.listdir(dossier_csv):
        if fichier.endswith(".csv"):
            df = pd.read_csv(os.path.join(dossier_csv, fichier))
            for classe in classes_recherchees:
                df_classe = df[df['class'] == classe]
                if not df_classe.empty:
                    # Extraire les coordonnées des boxes et calculer les moyennes et variances
                    positions_x = []
                    positions_y = []
                    for _, row in df_classe.iterrows():
                        box = extract_box_coordinates(row['box'])
                        positions_x.append(box[0] + box[2] / 2)  # Calcul du centre X
                        positions_y.append(box[1] + box[3] / 2)  # Calcul du centre Y
                    positions_moyennes_x[classe].append(np.mean(positions_x))
                    positions_moyennes_y[classe].append(np.mean(positions_y))
                    variances_x[classe].append(np.var(positions_x))
                    variances_y[classe].append(np.var(positions_y))
                else:
                    positions_moyennes_x[classe].append(0)
                    positions_moyennes_y[classe].append(0)
                    variances_x[classe].append(0)
                    variances_y[classe].append(0)

    # Calcul de la moyenne et de la variance pour chaque classe
    resultats_x = {classe: {'mean': np.mean(positions_moyennes_x[classe]), 'variance': np.mean(variances_x[classe])} for classe in classes_recherchees}
    resultats_y = {classe: {'mean': np.mean(positions_moyennes_y[classe]), 'variance': np.mean(variances_y[classe])} for classe in classes_recherchees}
    
    return resultats_x, resultats_y

# Analyser les deux dossiers
positions_x_1, positions_y_1 = analyser_position_moyenne(dossier_csv_1)
positions_x_2, positions_y_2 = analyser_position_moyenne(dossier_csv_2)

# Fonction pour afficher les positions sous forme d'image avec des cercles
def create_position_image_with_circles(positions_x, positions_y, variances_x, variances_y, title="Position"):
    image = np.ones((500, 500, 3))  # Image blanche
    
    for classe in classes_recherchees:
        x = int(positions_x[classe]['mean'])
        y = int(positions_y[classe]['mean'])
        variance_x = np.sqrt(variances_x[classe]['mean']) * 20
        variance_y = np.sqrt(variances_y[classe]['mean']) * 20
        color = couleurs_classes[classe]
        
        cv2.ellipse(image, (x, y), (int(variance_x), int(variance_y)), 0, 0, 360, color, 2)
        cv2.putText(image, f'{classe}', (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    
    cv2.putText(image, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    return image

# Créer les images pour chaque dossier
image_1 = create_position_image_with_circles(positions_x_1, positions_y_1, positions_x_1, positions_y_1, "Alice 6 mois")
image_2 = create_position_image_with_circles(positions_x_2, positions_y_2, positions_x_2, positions_y_2, "Alice 12 mois")

# Affichage dans Streamlit
st.title("Analyse des Positions")

# Affichage des images avec axes et légendes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.imshow(image_1)
ax1.set_xlim([0, 500])
ax1.set_ylim([500, 0])
ax1.set_xlabel("Position X")
ax1.set_ylabel("Position Y")

ax2.imshow(image_2)
ax2.set_xlim([0, 500])
ax2.set_ylim([500, 0])
ax2.set_xlabel("Position X")
ax2.set_ylabel("Position Y")

st.pyplot(fig)

# Création d'un tableau pour les résultats
df_resultats = pd.DataFrame({
    'Classe': classes_recherchees,
    'Moyenne X (Dossier 1)': [f"{positions_x_1[c]['mean']:.2f}" for c in classes_recherchees],
    'Variance X (Dossier 1)': [f"{positions_x_1[c]['variance']:.2f}" for c in classes_recherchees],
    'Moyenne Y (Dossier 1)': [f"{positions_y_1[c]['mean']:.2f}" for c in classes_recherchees],
    'Variance Y (Dossier 1)': [f"{positions_y_1[c]['variance']:.2f}" for c in classes_recherchees],
    'Moyenne X (Dossier 2)': [f"{positions_x_2[c]['mean']:.2f}" for c in classes_recherchees],
    'Variance X (Dossier 2)': [f"{positions_x_2[c]['variance']:.2f}" for c in classes_recherchees],
    'Moyenne Y (Dossier 2)': [f"{positions_y_2[c]['mean']:.2f}" for c in classes_recherchees],
    'Variance Y (Dossier 2)': [f"{positions_y_2[c]['variance']:.2f}" for c in classes_recherchees]
})

st.write("### Tableau des moyennes et variances")
st.dataframe(df_resultats, hide_index=True)
