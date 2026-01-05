import streamlit as st
import torch
import fasttext
import os
import pandas as pd
import torch.nn as nn
from difflib import get_close_matches

# ============================================
# 1. CONFIGURATION DES CHEMINS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Chemins des modèles
FASTTEXT_PATH = os.path.join(BASE_DIR, "modelsfastext", "cc.fr.300.bin")
MODEL_PATH = os.path.join(BASE_DIR, "models", "citp_classifier_model.pth")

# --- VOS DEUX SOURCES DE DONNÉES ---
# 1. Le référentiel officiel (celui que vous venez de donner)
ISCO_REF_PATH = r"C:\Users\Sy Savane Idriss\isco-classifier-web\data\CITP_08.xlsx"

# 2. Les données d'entraînement (pour les suggestions de correction)
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "entrainer2_propre.xlsx")

# ============================================
# 2. CHARGEMENT ET CACHE
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
    ft = fasttext.load_model(FASTTEXT_PATH)
    checkpoint = torch.load(MODEL_PATH, weights_only=False)
    model = CITPClassifier(300, checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return ft, model, checkpoint['label_encoder']

@st.cache_data
def load_data():
    # Chargement du référentiel officiel (CITP_08)
    # Note : Vérifiez si vos colonnes s'appellent 'nomenclature' et 'code' dans CITP_08.xlsx
    df_ref = pd.read_excel(ISCO_REF_PATH)
    mapping_officiel = pd.Series(df_ref.code.values, index=df_ref.nomenclature).to_dict()
    
    # Chargement des noms de métiers d'entraînement (pour aider l'IA)
    df_train = pd.read_excel(TRAIN_DATA_PATH)
    list_train = df_train['nomenclature'].unique().tolist()
    
    return mapping_officiel, list_train

# ============================================
# 3. INTERFACE UTILISATEUR
# ============================================
st.set_page_config(page_title="ISCO Expert System", page_icon="💼", layout="wide")

st.title("💼 Système Expert de Classification ISCO-08")
st.markdown("---")

# Chargement
ft_model, classifier, le = load_ai_models()
isco_mapping, training_jobs = load_data()
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
elif free_text:
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