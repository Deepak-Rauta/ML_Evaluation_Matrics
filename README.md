Machine Learning Model Evaluation Metrics
This repository demonstrates the evaluation of five machine learning models using Object-Oriented Programming (OOP). Each model is trained, evaluated, and saved as a .pkl (pickle) file for future use. Evaluation metrics such as precision, recall, F1-score, R² score, confusion matrix, and ROC curve are used to assess and compare the models' performances.

Project Overview
In this project, the following machine learning models were developed and evaluated:

Decision Tree (DTree)
K-Nearest Neighbors (KNN)
Random Forest
Naive Bayes
Logistic Regression
For each model, the following evaluation metrics were calculated:

Precision
Recall
F1-Score
R² Score
Confusion Matrix
ROC Curve
The trained models are saved as .pkl files using the pickle module.

Project Structure
ML_Evaluation_Matrics/
├── main.py
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── random_forest.pkl
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
├── app.py
├── README.md
└── requirements.txt

Confusion Matrix Visualization
The confusion matrices for each model are visualized and saved in the project. You can find the generated images for confusion matrices in the project directory (e.g., confusion_matrix_dtree.png, confusion_matrix_knn.png, etc.).

ROC Curve Visualization
Similarly, the ROC curves for each model are plotted and saved as images (e.g., roc_curve_dtree.png, roc_curve_knn.png, etc.).
