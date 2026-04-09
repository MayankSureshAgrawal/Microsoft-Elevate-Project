import os
import pickle
import streamlit as st

# ───────────────────────────────────────
# PAGE CONFIG
# ───────────────────────────────────────
st.set_page_config(
    page_title="MedAI Lite",
    page_icon="🩺",
    layout="centered"
)

# ───────────────────────────────────────
# CUSTOM CSS (PREMIUM UI)
# ───────────────────────────────────────
st.markdown("""
<style>

/* Title */
.main-title {
    text-align:center;
    font-size:3.2rem;
    font-weight:800;
    color:#0f172a;
}

/* Subtitle */
.subtitle {
    text-align:center;
    font-size:1.2rem;
    color:#64748b;
    margin-bottom:2rem;
}

/* Sidebar Card Buttons */
.nav-card {
    background: #f1f5f9;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
    text-align: center;
    font-weight: 600;
    cursor: pointer;
}

.nav-card:hover {
    background: #e2e8f0;
}

/* Buttons */
.stButton>button {
    width: 100%;
    height: 3em;
    font-size: 1.1rem;
    border-radius: 8px;
    background-color:#0891b2;
    color:white;
}

/* Headers */
h1, h2, h3 {
    font-weight:700;
}

</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────
# HEADER
# ───────────────────────────────────────
st.markdown("<div class='main-title'>🩺 MedAI Lite</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Based Disease Prediction System</div>", unsafe_allow_html=True)

# ───────────────────────────────────────
# LOAD MODELS
# ───────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    models = {}
    models['diabetes'] = pickle.load(open(os.path.join(BASE_DIR, "Saved_Models/diabetes_model.sav"), "rb"))
    models['heart'] = pickle.load(open(os.path.join(BASE_DIR, "Saved_Models/heart_disease_model.sav"), "rb"))
    models['liver'] = pickle.load(open(os.path.join(BASE_DIR, "Saved_Models/liver_disease_model.sav"), "rb"))
    return models

models = load_models()

# ───────────────────────────────────────
# SIDEBAR (CARD STYLE NAVIGATION)
# ───────────────────────────────────────
st.sidebar.title("🔍 Select Model")

if "page" not in st.session_state:
    st.session_state.page = "Diabetes"

if st.sidebar.button("🩸 Diabetes Prediction"):
    st.session_state.page = "Diabetes"

if st.sidebar.button("❤️ Heart Disease Prediction"):
    st.session_state.page = "Heart Disease"

if st.sidebar.button("🫀 Liver Disease Prediction"):
    st.session_state.page = "Liver Disease"

page = st.session_state.page

# ───────────────────────────────────────
# DIABETES
# ───────────────────────────────────────
if page == "Diabetes":
    st.header("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 200, 110)
        bp = st.number_input("Blood Pressure", 0, 150, 70)
        skin = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("Predict Diabetes"):
        result = models['diabetes'].predict([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        if result[0] == 1:
            st.error("⚠️ High Risk of Diabetes")
        else:
            st.success("✅ Low Risk of Diabetes")

# ───────────────────────────────────────
# HEART
# ───────────────────────────────────────
if page == "Heart Disease":
    st.header("❤️ Heart Disease Prediction")

    age = st.number_input("Age", 1, 120, 50)
    sex = st.selectbox("Sex (1=Male, 0=Female)", [0, 1])
    cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 1)
    trestbps = st.number_input("Resting BP", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 200)

    if st.button("Predict Heart Disease"):
        input_data = [age, sex, cp, trestbps, chol, 0, 0, 150, 0, 1.0, 1, 0, 1]
        result = models['heart'].predict([input_data])

        if result[0] == 1:
            st.error("⚠️ Heart Disease Risk Detected")
        else:
            st.success("✅ No Heart Disease Risk")

# ───────────────────────────────────────
# LIVER
# ───────────────────────────────────────
if page == "Liver Disease":
    st.header("🫀 Liver Disease Prediction")

    age = st.number_input("Age", 1, 100, 45)
    gender = st.selectbox("Gender (1=Male, 0=Female)", [0, 1])
    tb = st.number_input("Total Bilirubin", 0.0, 75.0, 1.0)
    db = st.number_input("Direct Bilirubin", 0.0, 20.0, 0.3)
    alk = st.number_input("Alkaline Phosphotase", 50, 2000, 300)

    if st.button("Predict Liver Disease"):
        input_data = [age, gender, tb, db, alk, 25, 35, 6.8, 3.3, 1.0]
        result = models['liver'].predict([input_data])

        if result[0] == 1:
            st.error("⚠️ Liver Disease Risk Detected")
        else:
            st.success("✅ No Liver Disease Risk")

# ───────────────────────────────────────
# FOOTER
# ───────────────────────────────────────
st.markdown("---")
st.markdown("Thank You !")