# Phishing Website Detection using Machine Learning

## Project Overview

This repository presents a reproducible machine learning pipeline for phishing website detection using engineered URL and website features.

The project reproduces and extends a publicly available phishing website detection study by organizing the workflow into a structured machine learning pipeline. The notebook includes exploratory data analysis, data preprocessing, model training, model evaluation, feature importance analysis, error analysis, comparison with the original implementation, and critical discussion.

Three classical machine learning models are trained and evaluated on a held-out test set to identify the most effective classifier for phishing website detection.

The project is based on the public repository:

https://github.com/gangeshbaskerr/Phishing-Website-Detection

---

## Dataset

The project uses the dataset:

```
DataFiles/5.urldata.csv
```

Each website record contains:

- Domain name
- URL-based features
- Website behavior features
- Security-related features
- Binary class label:
  - **0 = Legitimate**
  - **1 = Phishing**

The `Domain` column is excluded during model training to prevent the models from memorizing specific website names. Instead, the models learn general phishing-related patterns from the engineered features.

---

## Repository Structure

```
cyber-phishing-review/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── figures/
└── src/
    ├── data.py
    ├── evaluation.py
    ├── models.py
    └── plots.py
```

---

## Technologies

- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Machine Learning Workflow

The notebook follows the complete machine learning workflow:

1. Load the dataset
2. Inspect and validate the data
3. Perform Exploratory Data Analysis (EDA)
4. Prepare the features for machine learning
5. Split the data into training and testing sets
6. Train multiple machine learning models
7. Evaluate and compare model performance
8. Analyze feature importance
9. Perform error analysis
10. Compare the results with the original implementation
11. Critically evaluate the methodology
12. Draw conclusions

---

## Models Evaluated

The following classification models are implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Main Results

The trained models are evaluated on a held-out test dataset to estimate their ability to generalize to previously unseen websites.

The notebook includes:

- Performance comparison table
- Confusion matrices
- ROC curves
- Feature importance analysis
- Error analysis
- Comparison with the original implementation
- Critical evaluation

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open `notebook.ipynb` using VS Code or Jupyter Notebook and execute all cells sequentially from top to bottom.

Example:

```bash
jupyter notebook
```

---

## References

- Original repository  
  https://github.com/gangeshbaskerr/Phishing-Website-Detection

- Scikit-learn Documentation  
  https://scikit-learn.org/

- Pandas Documentation  
  https://pandas.pydata.org/