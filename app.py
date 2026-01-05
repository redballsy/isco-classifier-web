import streamlit as st
import pandas as pd
import fasttext
import torch
import torch.nn as nn
import os
import gc

# Configuration de la page
st.set_page_config(page_title="Classifieur ISCO", layout="centered")

# Chemins des fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelsfastext")
DATA_DIR = os.path.join(BASE_DIR, "data")

FASTTEXT_PATH = os.path.join(MODEL_DIR, "cc.fr.300.ftz")
PYTORCH_PATH = os.path.join(MODEL_DIR, "fasttext_citp_v1.pt")
EXCEL_PATH = os.path.join(DATA_DIR, "CITP_08.xlsx")

# --- CHARGEMENT DES MODÈLES AVEC MISE EN CACHE ---
@st.cache_resource
def load_all_resources():
    st.write("⏳ Chargement des modèles en cours...")
    
    # 1. Chargement FastText
    ft_model = fasttext.load_model(FASTTEXT_PATH)
    
    # 2. Chargement PyTorch (Mode CPU obligatoire pour Streamlit Cloud)
    # Note : On suppose que ton modèle est une instance de nn.Module sauvegardée
    pt_model = torch.load(PYTORCH_PATH, map_location=torch.device('cpu'))
    pt_model.eval()
    
    # 3. Chargement Excel
    df_ref = pd.read_excel(EXCEL_PATH)
    
    # Nettoyage mémoire
    gc.collect()
    
    return ft_model, pt_model, df_ref

# Initialisation
try:
    ft_model, pt_model, df_ref = load_all_resources()
    st.success("✅ Modèles et référentiel chargés !")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement : {e}")
    st.stop()

# --- INTERFACE UTILISATEUR ---
st.title("Search ISCO Classifier")
st.subheader("Classification automatique des métiers (CITP-08)")

job_description = st.text_input("Entrez l'intitulé du métier :", placeholder="Ex: Développeur logiciel")

if job_description:
    # 1. Prétraitement / Embedding avec FastText
    # On récupère le vecteur de la phrase (sentence vector)
    sentence_vector = ft_model.get_sentence_vector(job_description)
    input_tensor = torch.tensor(sentence_vector).unsqueeze(0) # Ajout dimension batch

    # 2. Prédiction avec PyTorch
    with torch.no_grad():
        output = pt_model(input_tensor)
        # On récupère l'index de la classe avec la plus haute probabilité
        _, predicted_idx = torch.max(output, 1)
        prediction = predicted_idx.item()

    # 3. Correspondance avec le fichier Excel
    # On suppose que ton modèle prédit un code qui correspond à une colonne 'Code' dans l'Excel
    resultat = df_ref[df_ref['Code'] == prediction]

    if not resultat.empty:
        st.write("### Résultat de la classification :")
        st.table(resultat)
    else:
        st.warning(f"Code prédit : {prediction}, mais aucune correspondance trouvée dans l'Excel.")

# Pied de page
st.markdown("---")
st.caption("Application propulsée par FastText, PyTorch et Streamlit")