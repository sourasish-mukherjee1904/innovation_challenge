import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.ensemble import RandomForestClassifier

# Path to the Kaggle dataset
path = r"C:\Users\sayak\.cache\kagglehub\datasets\dhivyeshrk\diseases-and-symptoms-dataset\versions\1"
csv_file = os.path.join(path, "Final_Augmented_dataset_Diseases_and_Symptoms.csv")

if not os.path.exists(csv_file):
    print("Error: CSV dataset not found.")
    sys.exit(1)

print("Loading dataset...")
df = pd.read_csv(csv_file)

# 12 Target Classes
target_classes = [
    'Blood Disorder', 'Chronic Kidney Disease', 'Chronic Respiratory Disease',
    'Diabetes', 'Diarrheal Disease', 'Hypertension', 'Leptospirosis',
    'Liver Infection', 'Mental Health Disorder', 'Mosquito-borne Fever',
    'Respiratory Infection', 'Typhoid'
]

# Disease Mappings from Kaggle diseases to our 12 classes
disease_mapping = {
    # Respiratory Infection
    "common cold": "Respiratory Infection",
    "flu": "Respiratory Infection",
    "pneumonia": "Respiratory Infection",
    "acute bronchitis": "Respiratory Infection",
    "acute bronchiolitis": "Respiratory Infection",
    
    # Diarrheal Disease
    "infectious gastroenteritis": "Diarrheal Disease",
    "noninfectious gastroenteritis": "Diarrheal Disease",
    "food poisoning": "Diarrheal Disease",
    
    # Chronic Respiratory Disease
    "asthma": "Chronic Respiratory Disease",
    "chronic obstructive pulmonary disease (copd)": "Chronic Respiratory Disease",
    
    # Diabetes
    "diabetes": "Diabetes",
    "gestational diabetes": "Diabetes",
    "diabetes insipidus": "Diabetes",
    
    # Hypertension
    "high blood pressure": "Hypertension",
    "pulmonary hypertension": "Hypertension",
    "malignant hypertension": "Hypertension",
    "kidney disease due to longstanding hypertension": "Hypertension",
    
    # Liver Infection
    "liver disease": "Liver Infection",
    
    # Mental Health Disorder
    "depression": "Mental Health Disorder",
    "anxiety": "Mental Health Disorder",
    "panic disorder": "Mental Health Disorder",
    "bipolar disorder": "Mental Health Disorder",
    
    # Blood Disorder
    "anemia": "Blood Disorder",
    "iron deficiency anemia": "Blood Disorder",
    "sickle cell anemia": "Blood Disorder",
    
    # Chronic Kidney Disease
    "chronic kidney disease": "Chronic Kidney Disease",
    "kidney failure": "Chronic Kidney Disease",
    "diabetic kidney disease": "Chronic Kidney Disease",
    
    # Mosquito-borne Fever
    "malaria": "Mosquito-borne Fever",
    "dengue fever": "Mosquito-borne Fever",
    
    # Typhoid
    "typhoid fever": "Typhoid",
}

# Category Mappings (Hierarchical Stacking domains)
category_mapping = {
    "Mosquito-borne Fever": 0, "Leptospirosis": 0, "Typhoid": 0,                # Category 0: acute_infectious
    "Hypertension": 1, "Chronic Kidney Disease": 1, "Diabetes": 1,                # Category 1: cardiorenal_metabolic
    "Respiratory Infection": 2, "Chronic Respiratory Disease": 2, "Diarrheal Disease": 2, "Liver Infection": 2, # Category 2: respiratory_digestive
    "Blood Disorder": 3, "Mental Health Disorder": 3                             # Category 3: systemic_psychiatric
}

# Filter rows that match our target classes
df_mapped = df[df["diseases"].isin(disease_mapping.keys())].copy()
df_mapped["target_disease"] = df_mapped["diseases"].map(disease_mapping)
df_mapped["target_category"] = df_mapped["target_disease"].map(category_mapping)

print(f"Mapped dataset size: {df_mapped.shape[0]} rows.")

# Map Kaggle symptom columns to our 44 symptoms
symptom_mapping = {
    "abdominal pain": ["sharp abdominal pain", "lower abdominal pain", "burning abdominal pain", "upper abdominal pain"],
    "appetite changes": ["decreased appetite", "excessive appetite"],
    "blurred vision": ["diminished vision", "double vision", "spots or clouds in vision"],
    "chest pain": ["sharp chest pain", "burning chest pain"],
    "chills": ["chills"],
    "cough": ["cough", "coughing up sputum"],
    "cramps": ["cramps and spasms", "muscle cramps contractures or spasms", "arm cramps or spasms", "leg cramps or spasms"],
    "diarrhea": ["diarrhea"],
    "dizziness": ["dizziness"],
    "fatigue": ["fatigue"],
    "fever": ["fever"],
    "frequent urination": ["frequent urination"],
    "high fever": ["fever"],
    "irritability": ["premenstrual tension or irritability"],
    "joint pain": ["joint pain"],
    "muscle pain": ["muscle pain"],
    "nausea": ["nausea"],
    "rash": ["eyelid lesion or rash", "skin rash", "diaper rash"],
    "severe headache": ["headache"],
    "shortness of breath": ["shortness of breath"],
    "sore throat": ["sore throat"],
    "sweating": ["sweating"],
    "swelling": ["throat swelling", "skin swelling", "hand or finger swelling", "leg swelling", "foot or toe swelling", "joint swelling"],
    "swollen lymph nodes": ["swollen lymph nodes"],
    "vomiting": ["vomiting", "vomiting blood"],
    "weight loss": ["recent weight loss"],
    "wheezing": ["wheezing"],
    "excessive worry": ["anxiety and nervousness"],
    "loss of appetite": ["decreased appetite"],
    "loss of smell/taste": ["disturbance of smell or taste"],
    "numbness": ["loss of sensation"],
    "persistent sadness": ["depression", "depressive or psychotic symptoms"],
    "sleep disturbance": ["insomnia", "sleepiness"],
    "delayed growth": ["lack of growth"],
}

our_symptoms = [
    'abdominal pain', 'anemia', 'appetite changes', 'blurred vision', 'body ache', 
    'bone deformities', 'chest pain', 'chills', 'cough', 'cramps', 
    'dehydration', 'delayed growth', 'diarrhea', 'dizziness', 'excessive worry', 
    'fatigue', 'fever', 'frequent infections', 'frequent urination', 'high fever', 
    'irritability', 'joint pain', 'loss of appetite', 'loss of interest', 'loss of smell/taste', 
    'muscle pain', 'nausea', 'numbness', 'pain episodes', 'pale skin', 
    'panic attacks', 'persistent sadness', 'rash', 'severe headache', 'shortness of breath', 
    'sleep disturbance', 'sore throat', 'sweating', 'swelling', 'swollen lymph nodes', 
    'unexplained weight loss', 'vomiting', 'weight loss', 'wheezing'
]

print("Preparing symptom feature columns...")
X_symptoms = pd.DataFrame(0, index=df_mapped.index, columns=our_symptoms)

for our_sym in our_symptoms:
    kag_cols = symptom_mapping.get(our_sym, [])
    for col in kag_cols:
        if col in df_mapped.columns:
            X_symptoms[our_sym] = X_symptoms[our_sym] | df_mapped[col]

# Generate dummy demographic columns (since Kaggle data doesn't have them, we keep them constant/dummy so model is compatible)
print("Adding demographic placeholders...")
X_demographics = pd.DataFrame({
    "age": 30,
    "gender": "F",
    "season": "Summer",
    "comorbidity": "nan",
    "smoking_status": "Never",
    "alcohol_use": "nan",
    "bmi": 22.0
}, index=df_mapped.index)

# One-hot encode demographic categories exactly as predictor.py prepare_input expects
# In prepare_input, it takes demographic values directly. However, CatBoost Classifier handles categorical features natively.
# For our RandomForest, we will fit ONLY on the 44 symptoms features (since demographics are not available in Kaggle dataset anyway, 
# and keeping them constant would mean their information gain is 0). 
# This means our hierarchical model will predict diseases based on symptoms!

X = X_symptoms.values
y_disease = df_mapped["target_disease"].values
y_category = df_mapped["target_category"].values

print("Training Stage 1 Classifier (Category Predictor)...")
stage1_model = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
stage1_model.fit(X, y_category)

print("Training Stage 2 Classifiers (Disease Predictors per Category)...")
stage2_models = {}
for cat_id in [0, 1, 2, 3]:
    cat_mask = (y_category == cat_id)
    X_cat = X[cat_mask]
    y_cat = y_disease[cat_mask]
    
    unique_classes = np.unique(y_cat)
    print(f"  Category {cat_id} has classes: {list(unique_classes)}")
    
    stage2_model = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    stage2_model.fit(X_cat, y_cat)
    stage2_models[cat_id] = stage2_model

# Ensure folder exists
os.makedirs("models/hierarchical", exist_ok=True)

# Save models
with open("models/hierarchical/stage1_model.pkl", "wb") as f:
    pickle.dump(stage1_model, f)

with open("models/hierarchical/stage2_models.pkl", "wb") as f:
    pickle.dump(stage2_models, f)

print("Hierarchical model trained and saved successfully!")
