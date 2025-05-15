# Student Performance Prediction App

This app predicts whether a student will **pass** or **fail** based on various features like parental education, gender, ethnicity, lunch type, test preparation course, and scores in math, reading, and writing.

The model is built using a machine learning pipeline that handles preprocessing and classification, then deployed as a web app using **Streamlit**.

---

## Features

- Input student details via a simple web interface.
- Pipeline automatically preprocesses categorical and numerical features.
- Predict pass/fail outcome using a trained decision tree classifier.
- View prediction results instantly.

---

## Pipeline Details

The pipeline includes the following steps:

1. **Preprocessing** (`preprocessor`):
   - **One-Hot Encoding** of categorical features:
     - `parental level of education`
     - `gender`
     - `race/ethnicity`
     - `lunch`
     - `test preparation course`
   - **Standard Scaling** of numerical features:
     - `math score`
     - `reading score`
     - `writing score`

2. **Model**:
   - **Decision Tree Classifier** with tuned hyperparameters:
     - `max_depth`: 10
     - `criterion`: 'entropy'

This pipeline ensures the data is prepared consistently during training and prediction.

---

## Using joblib for Saving and Loading

To save and load the trained model pipeline efficiently, this project uses **joblib**, a Python library designed for serializing Python objects like machine learning models.

- **Saving the pipeline:**

  ```python
  import joblib

  joblib.dump(grid_search.best_estimator_, 'best_pipeline.pkl')
