import streamlit as st
import pandas as pd
import fasttext
import torch
import os
import gc

st.set_page_config(page_title="ISCO Classifier", layout="wide")

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASTTEXT_PATH = os.path.join(BASE_DIR, "modelsfastext", "cc.fr.300.ftz")
PYTORCH_PATH = os.path.join(BASE_DIR, "modelsfastext", "fasttext_citp_v1.pt")
EXCEL_PATH = os.path.join(BASE_DIR, "data", "CITP_08.xlsx")

# Utilisation de @st.cache_resource avec une limite pour éviter les fuites
@st.cache_resource(show_spinner="Chargement du cerveau de l'IA...")
def load_models():
    # On ne charge FastText que si nécessaire
    ft = fasttext.load_model(FASTTEXT_PATH)
    # Chargement léger du modèle PT
    pt = torch.load(PYTORCH_PATH, map_location="cpu")
    pt.eval()
    gc.collect()
    return ft, pt

@st.cache_data
def load_data():
    return pd.read_excel(EXCEL_PATH)

st.title("🔍 Classifieur de métiers ISCO")
st.write("Entrez un métier pour trouver son code CITP-08.")

# L'ASTUCE : On ne charge rien tant que l'utilisateur n'a pas interagi
if 'ready' not in st.session_state:
    st.session_state.ready = False

if not st.session_state.ready:
    if st.button("🚀 Démarrer l'application"):
        st.session_state.ready = True
        st.rerun()
else:
    try:
        ft_model, pt_model = load_models()
        df_ref = load_data()
        
        job_query = st.text_input("Intitulé du poste :", "")

        if job_query:
            # Prédiction
            vec = ft_model.get_sentence_vector(job_query)
            tensor = torch.tensor(vec).unsqueeze(0)
            
            with torch.no_grad():
                output = pt_model(tensor)
                prediction = torch.max(output, 1)[1].item()
            
            # Affichage
            # Vérifie bien que la colonne s'appelle 'Code' dans ton Excel
            res = df_ref[df_ref['Code'] == prediction]
            
            if not res.empty:
                st.success(f"Résultat trouvé pour le code : {prediction}")
                st.dataframe(res, use_container_width=True)
            else:
                st.warning(f"Code {prediction} prédit, mais non trouvé dans le fichier Excel.")
                
    except Exception as e:
        st.error(f"Erreur technique : {e}")
        if st.button("Réinitialiser"):
            st.cache_resource.clear()
            st.rerun()