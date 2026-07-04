# Phishing Website Detection using Machine Learning

## Project Overview

This project reproduces and extends a phishing website detection study using classical machine learning techniques.

The objective is to classify websites as **legitimate** or **phishing** based on manually engineered URL and website-related features. Multiple machine learning models are trained and compared using standard classification metrics to identify the most effective approach.

This work is based on the public repository:

https://github.com/gangeshbaskerr/Phishing-Website-Detection

while reorganizing the workflow into a reproducible machine learning pipeline with additional evaluation and analysis.

---

## Dataset

The project uses the dataset:

```
DataFiles/5.urldata.csv
```

Each website contains:

- Domain name
- URL-based features
- Website behavior features
- Security-related features
- Binary label:
  - **0 = Legitimate**
  - **1 = Phishing**

The `Domain` column is excluded during training to prevent the models from memorizing specific website names.

---

## Repository Structure

```
cyber-phishing-review/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── figures/
├── src/
└── data/
```

---

## Machine Learning Workflow

The notebook follows the following pipeline:

1. Load the dataset
2. Inspect and validate the data
3. Exploratory Data Analysis (EDA)
4. Feature engineering and preprocessing
5. Train/Test split
6. Model training
7. Model evaluation
8. Feature importance analysis
9. Error analysis
10. Comparison with the original implementation
11. Critical evaluation
12. Conclusions

---

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Main Results

The trained models are compared using a held-out test dataset.

The notebook includes:

- Performance comparison table
- Confusion matrices
- ROC curves
- Feature importance analysis
- Error analysis
- Comparison with the original implementation

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open the notebook and execute all cells in order.

```bash
jupyter notebook
```

or open the notebook directly in VS Code.

---

## References

- Original repository:
  https://github.com/gangeshbaskerr/Phishing-Website-Detection

- Scikit-learn Documentation:
  https://scikit-learn.org/

- Pandas Documentation:
  https://pandas.pydata.org/