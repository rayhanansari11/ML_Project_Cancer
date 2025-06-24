# 🎗️ Cancer Prediction App

This project is a **Streamlit web app** for predicting cancer risk using:
- 🧾 **Tabular model** — based on age, BMI, lifestyle, genetic risk, etc.
- 🩻 **CNN image model** — detects lung cancer types from chest X-ray images.

👉 [Live Demo (if hosted)](https://ml-project-cancer.streamlit.app/)

---

## 🚀 Features

✅ Predict cancer likelihood from:
- **Tabular data** (age, BMI, smoking, alcohol, etc.)
- **Chest X-ray images** (lung cancer types: adenocarcinoma, large cell carcinoma, squamous cell carcinoma, normal)

✅ Clean, user-friendly UI  
✅ PDF report download  
✅ Probability visualization (charts)  
✅ Educational content on cancer causes, prevention, and treatment  

---

## ⚙️ How to Run Locally

1️⃣ **Clone the repository**
```bash
git clone https://github.com/rayhanansari11/ML_Project_Cancer.git
cd ML_Project_Cancer

🧠 Models

Tabular model: Trained using scikit-learn (LogisticRegression, RandomForestClassifier, SVC, DecisionTreeClassifier, GradientBoostingClassifier)

Image model: CNN using transfer learning (Xception base + custom classifier layers)

ML_Project_Cancer/
├── app.py                  # Main Streamlit app
├── models/
│   ├── cancer_model.pkl     # Trained tabular model
│   └── keras_model.h5       # Trained CNN model
├── images/                  # Background and logo images
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation


👨‍💻 Authors
Rayhan Mahmud Ansari
Dept. of CSE, Sylhet Engineering College
GitHub | LinkedIn

Nurul Islam Opu
Dept. of CSE, Sylhet Engineering College
GitHub | LinkedIn

