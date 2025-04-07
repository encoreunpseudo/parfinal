import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Définition des dossiers CSV
dossier_csv_1 = "./csv_essai"
dossier_csv_2 = "./csv_essai"

# Configuration
fps = 30
classes_recherchees = ["mains", "visages", "P", "OP", "ONP", "ONG", "OAG"]

def analyser_temps_presence(dossier_csv):
    temps_moyens_par_fichier = []
    variances_temps_par_classe = {classe: [] for classe in classes_recherchees}

    for fichier in os.listdir(dossier_csv):
        if fichier.endswith(".csv"):
            df = pd.read_csv(os.path.join(dossier_csv, fichier))
            df = df.sort_values(by=['class', 'frame'])

            temps_moyens = {}
            for classe in classes_recherchees:
                df_classe = df[df['class'] == classe]
                if not df_classe.empty:
                    first_frame = df_classe['frame'].min()
                    last_frame = df_classe['frame'].max()
                    temps_presence = (last_frame - first_frame) / fps
                    variance = df_classe['frame'].var() / (fps ** 2)  # Variance en secondes²
                    variances_temps_par_classe[classe].append(variance)
                else:
                    temps_presence = 0
                    variances_temps_par_classe[classe].append(0)

                temps_moyens[classe] = temps_presence

            temps_moyens_par_fichier.append(temps_moyens)

    temps_moyens_global = {
        classe: sum(d[classe] for d in temps_moyens_par_fichier) / len(temps_moyens_par_fichier)
        for classe in classes_recherchees
    }

    variances_temps_globales = {
        classe: np.mean(variances_temps_par_classe[classe]) for classe in classes_recherchees
    }

    return temps_moyens_global, variances_temps_globales

st.title("Temps de présence moyen")

# Analyser les deux dossiers
temps_1, variance_1 = analyser_temps_presence(dossier_csv_1)
temps_2, variance_2 = analyser_temps_presence(dossier_csv_2)

# Créer un graphique
fig, ax = plt.subplots(figsize=(10, 6))

# Barres des temps de présence moyens
bar_width = 0.35
x = np.arange(len(classes_recherchees))

# Barres des temps de présence pour chaque dossier
ax.bar(x - bar_width / 2, [temps_1[c] for c in classes_recherchees], bar_width, 
       yerr=[np.sqrt(variance_1[c]) for c in classes_recherchees], capsize=5, label='Dossier 1', color='b', alpha=0.6)

ax.bar(x + bar_width / 2, [temps_2[c] for c in classes_recherchees], bar_width, 
       yerr=[np.sqrt(variance_2[c]) for c in classes_recherchees], capsize=5, label='Dossier 2', color='r', alpha=0.6)

# Ajouter les titres et labels
ax.set_xlabel('Classes')
ax.set_ylabel('Temps de présence moyen (en secondes)')
ax.set_title('Comparaison des temps de présence moyen entre les deux dossiers')

# Ajouter la légende
ax.legend()

# Ajuster les ticks des axes X
ax.set_xticks(x)
ax.set_xticklabels(classes_recherchees, rotation=45)

# Afficher le graphique
st.pyplot(fig)

# Créer un DataFrame pour le tableau des valeurs
df_resultats = pd.DataFrame({
    'Classe': classes_recherchees,
    'Moyenne (Dossier 1)': [f"{temps_1[c]:.2f}" for c in classes_recherchees],
    'Variance (Dossier 1)': [f"{variance_1[c]:.2f}" for c in classes_recherchees],
    'Moyenne (Dossier 2)': [f"{temps_2[c]:.2f}" for c in classes_recherchees],
    'Variance (Dossier 2)': [f"{variance_2[c]:.2f}" for c in classes_recherchees]
})

# Afficher le tableau sous le graphique
st.write("### Tableau des moyennes et variances")
st.dataframe(df_resultats, hide_index=True)
