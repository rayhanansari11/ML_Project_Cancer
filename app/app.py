import streamlit as st
import pandas as pd
import joblib
import base64
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from keras.models import load_model  # no standalone keras unless installed
from PIL import Image, ImageOps
import numpy as np

# === Load models ===
tabular_model = joblib.load("models/cancer_model.pkl")
cnn_model = load_model("models/keras_model.h5", compile=False)

# === Feature Names ===
feature_names = [
    "Age", "Gender", "BMI", "Smoking", "GeneticRisk",
    "PhysicalActivity", "AlcoholIntake", "CancerHistory"
]

# === Background Image Function ===
def set_bg_image(image_path):
    with open(image_path, "rb") as f:
        img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# Apply background
set_bg_image("images/background_image.jpg")

# === PDF Report Generator ===
def generate_pdf(data: dict, diagnosis: str, confidence: float):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 50, "Cancer Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 90, f"Prediction Result: {diagnosis}")
    c.drawString(40, height - 110, f"Confidence: {confidence:.2f}%")

    y = height - 150
    for k, v in data.items():
        c.drawString(40, y, f"{k}: {v}")
        y -= 20

    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Health Recommendation:")
    y -= 20
    c.setFont("Helvetica", 11)

    if diagnosis == "Cancer":
        suggestions = [
            "→ Please consult a specialist for further screening.",
            "→ Early diagnosis can lead to better outcomes.",
            "→ Maintain a healthy lifestyle and follow up regularly."
        ]
    else:
        suggestions = [
            "→ No signs detected, keep up with regular checkups.",
            "→ Continue a healthy lifestyle (exercise, avoid smoking/alcohol)."
        ]

    for tip in suggestions:
        c.drawString(50, y, tip)
        y -= 18

    c.save()
    buffer.seek(0)
    return buffer

# === Sidebar Navigation ===
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", [
    "Home", 
    "Cancer Prediction (Tabular)", 
    "Lung Cancer Prediction (Image CNN)", 
    "Learn More About Cancer", 
    "About"
])

# === Home Page ===
if page == "Home":
    st.title("Welcome to the Cancer Prediction App")
    st.write("""
        This app provides two way of cancer prediction:
        - **Tabular model** (age, BMI, lifestyle factors)
        - **CNN image model** (chest X-ray images)

        Use the sidebar to navigate and try both models.
    """)

# === Tabular Cancer Prediction ===
elif page == "Cancer Prediction (Tabular)":
    st.title("🩺 Cancer Prediction (Tabular Data)")
    st.subheader("Enter patient details")

    with st.expander("ℹ️ Dataset Structure Info"):
        st.markdown("""
        **Features:**  
        - `Age` : 20 - 80 (years)  
        - `Gender` : 0 = Male, 1 = Female  
        - `BMI` : 15.0 - 40.0  
        - `Smoking` : 0 = No, 1 = Yes  
        - `GeneticRisk` : 0 = Low, 1 = Medium, 2 = High  
        - `PhysicalActivity` : 0 - 10 (hours/week)  
        - `AlcoholIntake` : 0 - 5 (units/week)  
        - `CancerHistory` : 0 = No, 1 = Yes  
        """)

    default_values = {
        "Age": 45,
        "Gender": 1,
        "BMI": 26.0,
        "Smoking": 0,
        "GeneticRisk": 1,
        "PhysicalActivity": 4.5,
        "AlcoholIntake": 2.5,
        "CancerHistory": 0
    }

    # === Gender Selection with button-like radio ===
    st.markdown("**Gender:**")
    gender_col1, gender_col2 = st.columns(2)
    with gender_col1:
        if st.button("👨 Male"):
            gender_val = 0
    with gender_col2:
        if st.button("👩 Female"):
            gender_val = 1

    # Default to previous if no button press
    if 'gender_val' not in locals():
        gender_val = default_values["Gender"]

    # === Other Inputs ===
    age = st.number_input("Age", min_value=20, max_value=80, value=default_values["Age"], step=1)
    bmi = st.number_input("BMI", min_value=15.0, max_value=40.0, value=default_values["BMI"], step=0.1)
    smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1], index=default_values["Smoking"])
    genetic = st.selectbox("Genetic Risk (0=Low, 1=Medium, 2=High)", [0, 1, 2], index=default_values["GeneticRisk"])
    activity = st.slider("Physical Activity (hours/week)", 0.0, 10.0, value=default_values["PhysicalActivity"], step=0.5)
    alcohol = st.slider("Alcohol Intake (units/week)", 0.0, 5.0, value=default_values["AlcoholIntake"], step=0.5)
    history = st.selectbox("Cancer History (0=No, 1=Yes)", [0, 1], index=default_values["CancerHistory"])

    user_input = [age, gender_val, bmi, smoking, genetic, activity, alcohol, history]

    if st.button("🔮 Predict"):
        input_df = pd.DataFrame([user_input], columns=feature_names)
        prediction = tabular_model.predict(input_df)[0]
        probability = tabular_model.predict_proba(input_df)[0][prediction]
        label = "🛑 Cancer Detected" if prediction == 1 else "✅ No Cancer"

        st.success(f"**Prediction**: {label}")
        st.info(f"**Confidence**: {probability * 100:.2f}%")

        report = input_df.copy()
        report["Prediction"] = label
        report["Confidence (%)"] = round(probability * 100, 2)
        st.subheader("📋 Prediction Summary")
        st.dataframe(report)

        st.subheader("🔍 Cancer Likelihood")
        fig, ax = plt.subplots()
        ax.bar(["No Cancer", "Cancer"], tabular_model.predict_proba(input_df)[0], color=["green", "red"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        pdf_buffer = generate_pdf(
            dict(zip(feature_names, user_input)),
            label.replace("🛑 ", "").replace("✅ ", ""),
            probability * 100
        )

        st.download_button(
            label="📥 Download Prediction Report (PDF)",
            data=pdf_buffer,
            file_name="cancer_prediction_report.pdf",
            mime="application/pdf"
        )


# === CNN Image Prediction ===
elif page == "Lung Cancer Prediction (Image CNN)":
    st.title("📷 Lung Cancer Prediction Using CNN")
    st.write("""
        Upload a chest X-ray image. The CNN model will classify it into:
        - Adenocarcinoma
        - Large Cell Carcinoma
        - Normal
        - Squamous Cell Carcinoma
    """)

    uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_resized = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized).astype(np.float32)
        normalized_image_array = (img_array / 127.5) - 1
        img_array_exp = np.expand_dims(normalized_image_array, axis=0)

        prediction = cnn_model.predict(img_array_exp)
        predicted_class_idx = np.argmax(prediction[0])

        class_labels = [
            "adenocarcinoma",
            "large.cell.carcinoma",
            "normal",
            "squamous.cell.carcinoma"
        ]

        st.success(f"Prediction: **{class_labels[predicted_class_idx]}**")
        st.write(f"Confidence: **{prediction[0][predicted_class_idx] * 100:.2f}%**")

        fig, ax = plt.subplots()
        ax.bar(class_labels, prediction[0], color=["red", "orange", "green", "purple"])
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

# === Learn More ===
elif page == "Learn More About Cancer":
    st.title("📘 Learn More About Cancer")

    # Common Causes
    with st.expander("🔎 **Common Causes of Cancer**", expanded=True):
        st.info("""
        - Smoking and tobacco use  
        - Obesity and unhealthy diet  
        - Genetic mutations and family history  
        - Exposure to radiation or harmful chemicals  
        - Lack of physical activity  
        - Alcohol consumption  
        """)

    # Risk reduction
    with st.expander("💡 **How to Deal With or Reduce Risk**", expanded=False):
        st.success("""
        - Quit smoking and avoid secondhand smoke  
        - Maintain a healthy diet and weight  
        - Regular exercise and physical activity  
        - Routine medical checkups  
        - Limit alcohol consumption  
        - Protect yourself from the sun  
        """)

    # Bangladesh impact + stages side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌍 Cancer Impact in Bangladesh")
        st.warning("""
        - Cancer is a growing public health concern in Bangladesh.  
        - An estimated 150,000+ new cases are reported each year.  
        - Major challenges include late diagnosis and limited access to advanced treatment.  
        - Awareness, screening programs, and healthcare infrastructure are improving.  
        """)

    with col2:
        st.subheader("📊 Stages of Cancer")
        st.info("""
        - **Stage 0**: Abnormal cells, not yet cancer.  
        - **Stage I**: Small and localized tumor.  
        - **Stage II & III**: Larger tumors, possible spread to lymph nodes.  
        - **Stage IV**: Metastatic cancer, spread to other organs.  
        """)

    # Hospitals
    st.subheader("🏥 Top Hospitals in Bangladesh")
    st.write("""
    - National Institute of Cancer Research & Hospital (NICRH), Dhaka  
    - Delta Medical College & Hospital  
    - United Hospital, Dhaka  
    - Square Hospitals Ltd  
    - Evercare Hospital, Dhaka  
    """)

    st.subheader("✈️ Popular International Hospitals for Treatment")
    st.write("""
    - Tata Memorial Hospital, Mumbai, India  
    - Apollo Hospitals, Chennai, India  
    - Mount Elizabeth Hospital, Singapore  
    - Bumrungrad International Hospital, Thailand  
    - MD Anderson Cancer Center, Texas, USA  
    - Mayo Clinic, Rochester, USA  
    """)

    # YouTube awareness video
    st.subheader("🎥 Cancer Awareness Video")
    st.video("https://youtu.be/n8ioqyZu42w?si=rhtqSVNE9IWK-Rj3")


elif page == "About":
    from PIL import Image

    st.title("👨‍💻 About the App")

    # === Contributors ===
    st.subheader("👥 Created by")

    col1, col2 = st.columns(2)

    with col1:
        st.image("images/rayhan.jpg", width=150)  # replace with your image path
        st.markdown("### **Rayhan Mahmud Ansari**")
        st.write("📘 Dept. of CSE, Sylhet Engineering College")
        st.write("📧 rayhan_mahmud@sec.ac.bd")
        st.markdown("[🌐 GitHub](https://github.com/rayhanansari11) | [🔗 LinkedIn](https://www.linkedin.com/in/rayhan-mahmud-ansari-566d/)")

    with col2:
        st.image("images/opu.jpg", width=150)  # replace with your image path
        st.markdown("### **Nurul Islam Opu**")
        st.write("📘 Dept. of CSE, Sylhet Engineering College")
        st.write("📧 nurulislamopu1@gmail.com")
        st.markdown("[🌐 GitHub](https://github.com/Nurulislamopu) | [🔗 LinkedIn](https://www.linkedin.com/in/nurul-islam-opu-8669ba247/)")

    # === App Details ===
    st.markdown("---")
    st.subheader("📱 What This App Does")
    st.success("""
    This app provides two modes of cancer prediction:
    - 🧾 **Tabular model** using personal and lifestyle information
    - 🩻 **CNN image model** analyzing chest X-ray images
    """)

    st.info("🎯 Built using **machine learning** and **deep learning** with a focus on simplicity, reliability, and accessibility.")

    st.markdown("#### 🔍 Key Features")
    st.markdown("""
    - ✅ Clean and intuitive user interface  
    - 📥 PDF report generation with prediction summary  
    - 📊 Visual representation of prediction confidence  
    - 📚 Learn about causes, prevention, and treatment options  
    - 🌐 International and local hospital info  
    """)

    # === Model Info Expandable ===
    with st.expander("🧠 Model Details"):
        st.markdown("""
        - **Tabular Model**: Trained with logistic regression, random forest, and other classifiers.  
        - **Image Model**: CNN using **Xception** as base + custom dense layers.  
        - **Tools Used**: TensorFlow, scikit-learn, Streamlit, ReportLab, Matplotlib, etc.
        """)

    # === Version ===
    st.markdown("---")
    st.markdown("📦 **Model Version:** `v1.0.3`")
    st.caption("Last updated: June 2025")
