import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# Load dataset
df = pd.read_csv("data/diabetes.csv")

# -------------------------------
# PHASE 11 — Load Dataset
# -------------------------------

print("FIRST 5 ROWS OF DATASET")
print(df.head())

# -------------------------------
# PHASE 12 — Understand Dataset
# -------------------------------

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns)

print("\nDATASET INFO")
print(df.info())

print("\nSTATISTICAL SUMMARY")
print(df.describe())

# -------------------------------
# PHASE 13 — Dataset Features
# -------------------------------

print("\nFEATURE MEANINGS")

features = {
    "Pregnancies": "Number of pregnancies",
    "Glucose": "Glucose level",
    "BloodPressure": "Blood pressure",
    "SkinThickness": "Skin thickness",
    "Insulin": "Insulin level",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes genetic score",
    "Age": "Patient age",
    "Outcome": "0 = No Diabetes, 1 = Diabetes"
}

for key, value in features.items():
    print(f"{key} : {value}")

# -------------------------------
# PHASE 14 — Data Cleaning
# -------------------------------

print("\nMISSING VALUES")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nDATASET SHAPE AFTER REMOVING DUPLICATES")
print(df.shape)

# -------------------------------
# PHASE 15 — Replace Invalid Zero Values
# -------------------------------

cols = ['Glucose', 'BloodPressure', 'BMI']

for col in cols:
    df[col] = df[col].replace(0, df[col].mean())

print("\nZERO VALUES REPLACED SUCCESSFULLY")

print("\nUPDATED DATASET")
print(df.head())

# -------------------------------
# PHASE 16 — Exploratory Data Analysis
# -------------------------------

# Correlation Heatmap

print("\nSHOWING HEATMAP...")

sns.heatmap(df.corr(), annot=True)

plt.show()

# Diabetes Outcome Count

print("\nSHOWING DIABETES COUNT GRAPH...")

sns.countplot(x='Outcome', data=df)

plt.show()

# -------------------------------
# PHASE 17 — Prepare Data for ML
# -------------------------------

# Features (Input Data)
X = df.drop('Outcome', axis=1)

# Labels (Output Data)
y = df['Outcome']

print("\nFEATURE DATA (X)")
print(X.head())

print("\nLABEL DATA (y)")
print(y.head())

# -------------------------------
# PHASE 18 — Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTRAINING DATA SHAPE")
print(X_train.shape)

print("\nTESTING DATA SHAPE")
print(X_test.shape)

# -------------------------------
# PHASE 19 — Train ML Model
# -------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")

# -------------------------------
# PHASE 20 — Make Predictions
# -------------------------------

predictions = model.predict(X_test)

print("\nPREDICTIONS")
print(predictions)

# -------------------------------
# PHASE 21 — Evaluate Accuracy
# -------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY")
print(accuracy)

# -------------------------------
# PHASE 22 — Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print(cm)

# -------------------------------
# PHASE 22 — Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, predictions)

print("\nCONFUSION MATRIX")
print(cm)

# Heatmap Graph

sns.heatmap(cm, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()# -------------------------------
# PHASE 23 — Classification Report
# -------------------------------

report = classification_report(y_test, predictions)

print("\nCLASSIFICATION REPORT")
print(report)

# -------------------------------
# PHASE 24 — Save Model
# -------------------------------

joblib.dump(model, 'models/diabetes_model.pkl')

print("\nMODEL SAVED SUCCESSFULLY")

# -------------------------------
# PHASE 25 — Load Model & Predict
# -------------------------------

# Load saved model
loaded_model = joblib.load('models/diabetes_model.pkl')

# Sample patient data
sample_data = [[2, 120, 70, 20, 79, 25.0, 0.5, 33]]

# Make prediction
prediction = loaded_model.predict(sample_data)

print("\nPATIENT PREDICTION")
print(prediction)

# Show result
if prediction[0] == 1:
    print("Patient is likely Diabetic")
else:
    print("Patient is NOT Diabetic")

# -------------------------------
# PHASE 26 — User Input Prediction
# -------------------------------

print("\nENTER PATIENT DETAILS")

pregnancies = int(input("Pregnancies: "))
glucose = int(input("Glucose Level: "))
blood_pressure = int(input("Blood Pressure: "))
skin_thickness = int(input("Skin Thickness: "))
insulin = int(input("Insulin Level: "))
bmi = float(input("BMI: "))
dpf = float(input("Diabetes Pedigree Function: "))
age = int(input("Age: "))

# Create input data
user_data = [[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]]

# Predict
user_prediction = loaded_model.predict(user_data)

print("\nPREDICTION RESULT")

if user_prediction[0] == 1:
    print("Patient is likely Diabetic")
else:
    print("Patient is NOT Diabetic")
    
        