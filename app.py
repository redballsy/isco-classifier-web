import streamlit as st
import torch
import fasttext
import os
import pandas as pd
import torch.nn as nn
from difflib import get_close_matches

# ============================================
# 1. CONFIGURATION DES CHEMINS (UNIVERSELS)
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# On regroupe tout dans le dossier modelsfastext pour plus de simplicité
MODEL_DIR = os.path.join(BASE_DIR, "modelsfastext")

# À la ligne 18 environ
FASTTEXT_PATH = os.path.join(MODEL_DIR, "cc.fr.300.ftz")

# TON modèle classifieur (on utilise le nom exact de ton fichier)
MODEL_PATH = os.path.join(MODEL_DIR, "fasttext_citp_v1.pt")

# Chemins des données (Relatifs pour GitHub)
ISCO_REF_PATH = os.path.join(BASE_DIR, "data", "CITP_08.xlsx")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "entrainer2_propre.xlsx")

# ============================================
# 2. ARCHITECTURE ET CHARGEMENT
# ============================================
class CITPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CITPClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.network(x)

@st.cache_resource
def load_ai_models():
    if not os.path.exists(FASTTEXT_PATH):
        st.error(f"Fichier manquant : {FASTTEXT_PATH}")
        return None, None, None
        
    # Charger FastText
    ft = fasttext.load_model(FASTTEXT_PATH)
    
    # Charger PyTorch (on force le CPU pour le Cloud)
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model = CITPClassifier(300, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return ft, model, checkpoint['label_encoder']

@st.cache_data
def load_data():
    try:
        df_ref = pd.read_excel(ISCO_REF_PATH)
        mapping_officiel = pd.Series(df_ref.code.values, index=df_ref.nomenclature).to_dict()
        
        df_train = pd.read_excel(TRAIN_DATA_PATH)
        list_train = df_train['nomenclature'].unique().tolist()
        
        return mapping_officiel, list_train
    except Exception as e:
        st.error(f"Erreur Excel : {e}")
        return {}, []

# ============================================
# 3. INTERFACE UTILISATEUR
# ============================================
st.set_page_config(page_title="ISCO Expert System", page_icon="💼", layout="wide")
st.title("💼 Système Expert de Classification ISCO-08")

# Chargement sécurisé
ft_model, classifier, le = load_ai_models()
isco_mapping, training_jobs = load_data()

if isco_mapping and ft_model:
    official_jobs = sorted([str(k) for k in isco_mapping.keys()])
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 Référentiel Officiel")
        selected_job = st.selectbox("Métier officiel :", options=[""] + official_jobs)

    with col2:
        st.subheader("🤖 Intelligence Artificielle")
        free_text = st.text_input("Libellé libre :", placeholder="Ex: Développeur fullstack...")

    result_code = None
    source = ""

    if selected_job:
        result_code = isco_mapping[selected_job]
        source = "Source : Base officielle CITP-08"
    elif free_text:
        with torch.no_grad():
            vector = torch.FloatTensor(ft_model.get_sentence_vector(free_text.lower())).unsqueeze(0)
            output = classifier(vector)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            result_code = le.inverse_transform([idx.item()])[0]
            source = f"Source : IA (Confiance {conf.item()*100:.2f}%)"
        
        suggestions = get_close_matches(free_text, official_jobs, n=1, cutoff=0.6)
        if suggestions:
            st.info(f"💡 Suggestion : **{suggestions[0]}**")

    if result_code:
        st.markdown("---")
        st.metric("Code ISCO", result_code)
        st.caption(source)