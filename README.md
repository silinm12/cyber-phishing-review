# Phishing Website Detection using Machine Learning

## Project Overview

This repository presents a reproducible machine learning pipeline for phishing website detection using engineered URL and website features.

The project reproduces and extends a publicly available phishing website detection study by organizing the workflow into a structured machine learning pipeline. The notebook includes exploratory data analysis, data preprocessing, model training, comprehensive model evaluation, duplicate-row robustness analysis, grouped-by-domain validation, threshold-based risk analysis, feature importance analysis, error analysis, comparison with the original implementation, and critical discussion.

Three classical machine learning models are trained and compared using a comprehensive evaluation framework to assess their performance and robustness for phishing website detection.

The repository is fully self-contained and includes the dataset, notebook, project report, and supporting source files, allowing the entire project to be reproduced after installing the required dependencies.

The project is based on the public repository:

https://github.com/gangeshbaskerr/Phishing-Website-Detection

---

## Dataset

The project uses the dataset:

```text
data/5.urldata.csv
```

The dataset is **included in this repository** under the `data/` folder. No additional download is required.

The dataset originates from the following public repository:

https://github.com/gangeshbaskerr/Phishing-Website-Detection

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

```text
cyber-phishing-review/
│
├── notebook.ipynb
├── README.md
├── requirements.txt
├── cyber_phishing_report.pdf
├── data/
│   └── 5.urldata.csv
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

The notebook follows a complete machine learning workflow consisting of:

1. Load the dataset
2. Inspect and validate the data
3. Perform Exploratory Data Analysis (EDA)
4. Prepare the features for machine learning
5. Split the data into training and testing sets
6. Train multiple machine learning models
7. Evaluate and compare model performance
8. Perform duplicate-row robustness analysis
9. Perform grouped-by-domain validation
10. Analyze feature importance
11. Perform threshold and error analysis
12. Compare the results with the original implementation
13. Critically evaluate the methodology
14. Draw conclusions

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
- F2 Score
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- PR-AUC

The notebook also includes precision-recall curves, confusion matrices, and threshold analysis to evaluate the trade-off between reducing false negatives and increasing false positives in phishing detection.

---

## Main Results

The trained models are evaluated on a held-out test set to estimate their ability to generalize to previously unseen websites.

Beyond the standard evaluation, the project investigates how duplicate rows, domain leakage, and decision-threshold selection influence the reported performance.

The notebook includes:

- Performance comparison across multiple evaluation metrics
- Confusion matrices
- Precision-recall curves
- Duplicate-row robustness analysis
- Grouped-by-domain validation using GroupShuffleSplit
- Threshold and risk trade-off analysis
- Feature importance analysis
- Error analysis
- Comparison with the original implementation
- Critical evaluation

---

## Project Report

The final project report is available in this repository:

- **cyber_phishing_report.pdf**

The report includes:

- Executive Summary
- Summary of the original repository
- Critical evaluation
- Feature engineering analysis
- Reproducibility analysis
- Experimental results
- Summing It Up
- Conclusion

---

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The dataset is already included in the repository.

1. Clone the repository:

```bash
git clone https://github.com/silinm12/cyber-phishing-review
cd cyber-phishing-review
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook notebook.ipynb
```

4. Execute all notebook cells from top to bottom.

The accompanying project report (`cyber_phishing_report.pdf`) summarizes the methodology, experimental results, robustness analyses, and critical evaluation of the reproduction study.

---

## References

- Original repository  
  https://github.com/gangeshbaskerr/Phishing-Website-Detection

- Scikit-learn Documentation  
  https://scikit-learn.org/

- Pandas Documentation  
  https://pandas.pydata.org/