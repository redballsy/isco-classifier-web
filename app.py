import streamlit as st
import torch
import fasttext
import os
import pandas as pd
import torch.nn as nn
import urllib.request
from difflib import get_close_matches

# ============================================
# 1. CONFIGURATION DES CHEMINS (AUTO-ADAPTATIFS)
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier unique pour tes modèles
MODEL_DIR = os.path.join(BASE_DIR, "modelsfastext")

# 1. Le modèle de langue (Sera téléchargé automatiquement s'il manque)
FASTTEXT_PATH = os.path.join(MODEL_DIR, "cc.fr.300.ftz")

# 2. Ton classifieur entraîné (Doit être présent sur GitHub)
MODEL_PATH = os.path.join(MODEL_DIR, "citp_classifier_model.pth")

# Chemins des données Excel
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

def download_fasttext():
    """Télécharge le modèle FastText compressé s'il est absent du serveur"""
    if not os.path.exists(FASTTEXT_PATH):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.ftz"
        with st.spinner("Initialisation du moteur de langue (environ 400Mo)..."):
            urllib.request.urlretrieve(url, FASTTEXT_PATH)
        st.success("Moteur de langue prêt !")

@st.cache_resource
def load_ai_models():
    # 1. Télécharger si nécessaire
    download_fasttext()
    
    try:
        # 2. Charger le moteur FastText
        ft = fasttext.load_model(FASTTEXT_PATH)
        
        # 3. Charger ton classifieur sur CPU
        if not os.path.exists(MODEL_PATH):
            st.error(f"Fichier manquant : {MODEL_PATH}")
            return None, None, None
            
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model = CITPClassifier(300, checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return ft, model, checkpoint['label_encoder']
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return None, None, None

@st.cache_data
def load_data():
    try:
        df_ref = pd.read_excel(ISCO_REF_PATH)
        mapping_officiel = pd.Series(df_ref.code.values, index=df_ref.nomenclature).to_dict()
        
        df_train = pd.read_excel(TRAIN_DATA_PATH)
        list_train = df_train['nomenclature'].unique().tolist()
        
        return mapping_officiel, list_train
    except Exception as e:
        st.error(f"Erreur fichiers Excel : {e}")
        return {}, []

# ============================================
# 3. INTERFACE UTILISATEUR
# ============================================
st.set_page_config(page_title="ISCO Expert System", page_icon="💼", layout="wide")

st.title("💼 Système Expert de Classification ISCO-08")
st.markdown("---")

# Chargement des ressources
ft_model, classifier, le = load_ai_models()
isco_mapping, training_jobs = load_data()

if isco_mapping and ft_model:
    official_jobs = sorted([str(k) for k in isco_mapping.keys()])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 Référentiel Officiel")
        selected_job = st.selectbox(
            "Rechercher un métier officiel :",
            options=[""] + official_jobs,
            format_func=lambda x: "Choisir dans la liste..." if x == "" else x
        )

    with col2:
        st.subheader("🤖 Intelligence Artificielle")
        free_text = st.text_input(
            "Saisissez un libellé libre :",
            placeholder="Ex: Expert en cybersécurité..."
        )

    # --- LOGIQUE ---
    result_code = None
    source = ""
    confidence = 100.0

    if selected_job:
        result_code = isco_mapping[selected_job]
        source = "Source : Référentiel CITP-08"
    elif free_text:
        with torch.no_grad():
            vector = torch.FloatTensor(ft_model.get_sentence_vector(free_text.lower())).unsqueeze(0)
            output = classifier(vector)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
            result_code = le.inverse_transform([idx.item()])[0]
            confidence = conf.item() * 100
            source = f"Source : Prédiction IA (Confiance {confidence:.2f}%)"
        
        suggestions = get_close_matches(free_text, official_jobs, n=1, cutoff=0.6)
        if suggestions:
            st.info(f"💡 Métier officiel suggéré : **{suggestions[0]}**")

    # --- AFFICHAGE DU RÉSULTAT ---
    if result_code:
        st.markdown("---")
        st.metric("Code ISCO identifié", result_code)
        st.caption(source)
        if not selected_job and confidence < 50:
            st.warning("⚠️ Attention : Confiance faible. Vérifiez manuellement.")
else:
    st.info("Initialisation de l'application en cours...")