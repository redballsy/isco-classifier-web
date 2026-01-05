import streamlit as st
import torch
import fasttext
import os
import pandas as pd
import torch.nn as nn
import urllib.request
from difflib import get_close_matches

# ============================================
# 1. CONFIGURATION DES CHEMINS (RELATIFS)
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins des modèles
MODEL_DIR = os.path.join(BASE_DIR, "modelsfastext")
# Utilisation du modèle compressé .ftz pour éviter le plantage mémoire
FASTTEXT_PATH = os.path.join(MODEL_DIR, "cc.fr.300.ftz")
MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

# Chemins des données (Utilisation de BASE_DIR pour que ça marche partout)
ISCO_REF_PATH = os.path.join(BASE_DIR, "data", "CITP_08.xlsx")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "entrainer2_propre.xlsx")

# ============================================
# 2. FONCTIONS DE CHARGEMENT ET AUTO-DOWNLOAD
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
    """Télécharge le modèle FastText compressé s'il est absent (pour GitHub/Cloud)"""
    if not os.path.exists(FASTTEXT_PATH):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.ftz"
        with st.spinner("Téléchargement du modèle linguistique (environ 400Mo)..."):
            urllib.request.urlretrieve(url, FASTTEXT_PATH)
        st.success("Modèle téléchargé !")

@st.cache_resource
def load_ai_models():
    # S'assurer que FastText est là
    download_fasttext()
    
    try:
        # Charger FastText (version légère .ftz)
        ft = fasttext.load_model(FASTTEXT_PATH)
        
        # Charger PyTorch sur CPU (vital pour le Cloud)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        model = CITPClassifier(300, checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return ft, model, checkpoint['label_encoder']
    except Exception as e:
        st.error(f"Erreur de chargement des modèles : {e}")
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
        st.error(f"Erreur de chargement des fichiers Excel : {e}")
        return {}, []

# ============================================
# 3. INTERFACE UTILISATEUR
# ============================================
st.set_page_config(page_title="ISCO Expert System", page_icon="💼", layout="wide")

st.title("💼 Système Expert de Classification ISCO-08")
st.markdown("---")

# Initialisation
ft_model, classifier, le = load_ai_models()
isco_mapping, training_jobs = load_data()

if isco_mapping:
    official_jobs = sorted([str(k) for k in isco_mapping.keys()])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 Référentiel Officiel")
        selected_job = st.selectbox(
            "Sélectionnez un métier officiel (recherche exacte) :",
            options=[""] + official_jobs,
            format_func=lambda x: "Rechercher un métier..." if x == "" else x
        )

    with col2:
        st.subheader("🤖 Intelligence Artificielle")
        free_text = st.text_input(
            "Ou saisissez un libellé libre (prédiction) :",
            placeholder="Ex: Spécialiste cloud computing..."
        )

    # --- LOGIQUE DE TRAITEMENT ---
    result_code = None
    source = ""
    confidence = 100.0

    if selected_job:
        result_code = isco_mapping[selected_job]
        source = "Source : Base de données officielle CITP-08"
    elif free_text and ft_model:
        # Prédiction
        with torch.no_grad():
            vector = torch.FloatTensor(ft_model.get_sentence_vector(free_text.lower())).unsqueeze(0)
            output = classifier(vector)
            probs = torch.softmax(output, dim=1)
            conf, idx = torch.max(probs, 1)
            
            result_code = le.inverse_transform([idx.item()])[0]
            confidence = conf.item() * 100
            source = f"Source : Prédiction IA (Confiance {confidence:.2f}%)"
        
        # Suggestion
        suggestions = get_close_matches(free_text, official_jobs, n=1, cutoff=0.6)
        if suggestions:
            st.info(f"💡 Le métier officiel le plus proche est : **{suggestions[0]}**")

    # --- AFFICHAGE ---
    if result_code:
        st.markdown("---")
        st.metric("Code ISCO prédit / trouvé", result_code)
        st.caption(source)
        
        if not selected_job and confidence < 50:
            st.warning("⚠️ L'IA a un doute sur cette saisie. Vérifiez la correspondance.")
else:
    st.error("Impossible d'afficher l'interface : Fichiers de données manquants.")