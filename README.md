# Predicting High-Performing Students

# Project Goal
The goal of this project is to develop a machine learning model that can accurately predict whether a student will be a "high performer," defined as achieving a Grade Point Average (GPA) of 3.0 or higher. The model is intended to help identify students who may excel, allowing for early recognition or resource allocation.

# Project Methodology
This project was approached in distinct phases, from data preparation to final model 
selection.

# Data Preparation
The dataset used for this project, Student_performance_data.csv, was of high quality and required minimal preprocessing. It was analyzed and found to have no missing data, which eliminated the need for imputation techniques. Furthermore, all features were already in a numeric format, so no categorical encoding (like one-hot encoding) was necessary. This allowed for a direct transition to the modeling phases.
Phase 1: Initial Regression Analysis
The project initially aimed to predict the exact GPA for each student using various regression models.

# Objective: To predict a continuous GPA value.
Models Explored: [List the regression models you tried, e.g., Linear Regression, Ridge Regression, etc.]
Outcome: While this approach provided insights into the factors influencing GPA, it was determined that a more direct, actionable prediction was needed. The specific business question was not "What will the exact GPA be?" but rather "Is this student likely to be a high performer?".

# Phase 2: Reframing as a Classification Problem
To better address the project's goal, the problem was converted from regression to binary classification.
Feature Engineering: A new target column, is_high_performer, was created. This column was populated based on a simple rule:
If a student's GPA > 3.0, then is_high_performer = 1 (High Performer).
Otherwise, is_high_performer = 0 (Not a High Performer).
Classification Model Development: With the new binary target, several classification models were trained and evaluated to determine the best one for the task.
Models Tested: Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree, and Random Forest.
Final Model Selection: The Random Forest Classifier was chosen as the final model. It consistently demonstrated the highest performance, especially in providing a good balance of precision and recall, and its ensemble nature makes it more robust and less prone to overfitting than a single Decision Tree.
Final Model Performance
The final Random Forest model was evaluated on a held-out test set. Its performance is summarized below.


               precision    recall  f1-score   support

           0       0.99      1.00      0.99       414
           1       0.97      0.92      0.94        65

    accuracy                           0.99       479
   macro avg       0.98      0.96      0.97       479
weighted avg       0.99      0.99      0.99       479


Interpretation:
Overall Accuracy: The model is correct about 99% of the time.
Performance for "High Performers" (Class 1):
Precision (0.97): When the model predicts a student will be a high performer, it is correct 97% of the time.
Recall (0.92): The model successfully identifies 88% of all students who are actually high performers.
This indicates a strong, reliable model that is effective at both identifying high performers and avoiding false positives.

How to Use the Deployed Model
The final, trained model is saved in the file rf_student.joblib. To use it for making predictions on new student data, follow the instructions below.
1. Requirements
Ensure you have the necessary Python libraries installed:
scikit-learn, joblib, pandas
2. Required Input Features
The model requires a specific set of 14 features to make a prediction. All input values must be numeric. The new data must contain the following columns in the correct order:
StudentID, Age, Gender, Ethnicity, ParentalEducation, StudyTimeWeekly, Absences, Tutoring, ParentalSupport, Extracurricular, Sports, Music, Volunteering, GradeClass.
3. Prediction Script Example
The following Python script shows how to load the model and predict whether a new student will be a high performer.
Generated python
import joblib
import pandas as pd

# Define the model filename
MODEL_FILE = 'rf_student.joblib'

# Load the trained model from the file
model = joblib.load(MODEL_FILE)
print(f"Model '{MODEL_FILE}' loaded successfully.")

# --- Prepare New Student Data for Prediction ---
new_student_data = {
    'StudentID': [1001],
    'Age': [17],
    'Gender': [1],               # 1 = Male
    'Ethnicity': [0],             # 0 = Asian
    'ParentalEducation': [2],     # 2 = Bachelor's
    'StudyTimeWeekly': [15.5],
    'Absences': [2],
    'Tutoring': [1],              # 1 = Yes
    'ParentalSupport': [4],       # On a scale, e.g., 1-5
    'Extracurricular': [1],       # 1 = Yes
    'Sports': [0],                # 0 = No
    'Music': [1],                 # 1 = Yes
    'Volunteering': [0],          # 0 = No
    'GradeClass': [11]            # e.g., 11th Grade
}

# Convert the dictionary to a pandas DataFrame
new_student_df = pd.DataFrame(new_student_data)


# --- Make a Prediction ---
prediction = model.predict(new_student_df)
prediction_proba = model.predict_proba(new_student_df)


# --- Interpret and Display the Result ---
is_high_performer = prediction[0]
confidence_scores = prediction_proba[0]

print("\n--- Prediction Results ---")
print(f"Predicted Outcome: {'High Performer' if is_high_performer == 1 else 'Not a High Performer'}")
print(f"Confidence Score (Probability):")
print(f"  - Not a High Performer (0): {confidence_scores[0]:.2%}")
print(f"  - High Performer (1):     {confidence_scores[1]:.2%}")


Project Files
Student_model_reg_class.ipynb: The Jupyter Notebook containing all data exploration, preprocessing, and model training code.
Student_performance_data.csv: The raw dataset used for the project.
rf_student.joblib: The saved, final Random Forest classification model object.
README.md: This documentation file.