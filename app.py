import streamlit as st
import torch
import fasttext
import os
import pandas as pd
import torch.nn as nn
import urllib.request
from difflib import get_close_matches

# ============================================
# 1. CONFIGURATION DES CHEMINS
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelsfastext")

# Moteur de langue (Téléchargement auto)
FASTTEXT_PATH = os.path.join(MODEL_DIR, "cc.fr.300.ftz")

# Ton classifieur (Doit être dans modelsfastext sur GitHub)
MODEL_PATH = os.path.join(MODEL_DIR, "citp_classifier_model.pth")

# Données Excel
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
    """Télécharge le modèle avec une identité factice pour éviter les erreurs HTTP"""
    if not os.path.exists(FASTTEXT_PATH):
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        
        url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.ftz"
        
        try:
            with st.spinner("Téléchargement du moteur de langue (400 Mo)... Veuillez patienter."):
                # Cette partie simule un navigateur pour éviter le rejet du serveur (HTTPError)
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                with urllib.request.urlopen(req) as response, open(FASTTEXT_PATH, 'wb') as out_file:
                    out_file.write(response.read())
            st.success("Moteur de langue téléchargé avec succès !")
        except Exception as e:
            st.error(f"Erreur de téléchargement : {e}")
            st.info("Le serveur distant a refusé la connexion. Retentez dans quelques instants.")

@st.cache_resource
def load_ai_models():
    download_fasttext()
    
    if not os.path.exists(FASTTEXT_PATH):
        return None, None, None

    try:
        # Charger FastText
        ft = fasttext.load_model(FASTTEXT_PATH)
        
        # Charger le classifieur
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
            model = CITPClassifier(300, checkpoint['num_classes'])
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return ft, model, checkpoint['label_encoder']
        else:
            st.error(f"Fichier introuvable : {MODEL_PATH}")
            return ft, None, None
    except Exception as e:
        st.error(f"Erreur au chargement des modèles : {e}")
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
# 3. INTERFACE
# ============================================
st.set_page_config(page_title="ISCO Expert System", page_icon="💼", layout="wide")
st.title("💼 Système Expert de Classification ISCO-08")

ft_model, classifier, le = load_ai_models()
isco_mapping, training_jobs = load_data()

if isco_mapping and ft_model and classifier:
    official_jobs = sorted([str(k) for k in isco_mapping.keys()])
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📖 Référentiel Officiel")
        selected_job = st.selectbox("Choisir un métier :", options=[""] + official_jobs)

    with col2:
        st.subheader("🤖 Intelligence Artificielle")
        free_text = st.text_input("Saisissez un libellé libre :")

    result_code = None
    source = ""

    if selected_job:
        result_code = isco_mapping[selected_job]
        source = "Source : Référentiel Officiel"
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
else:
    st.warning("Chargement des composants en cours... Si cela prend plus de 5 minutes, vérifiez les fichiers sur GitHub.")