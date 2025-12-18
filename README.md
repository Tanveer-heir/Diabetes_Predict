# ============================================================
# 🟣 Project — Tree-Based Models & Hyperparameter Tuning
# ============================================================
#
# Project Focus
# -------------
# This project explores tree-based machine learning models and their ability
# to learn non-linear decision boundaries. The goal is to systematically improve
# model performance using ensemble methods and hyperparameter tuning.

# Dataset
# -------
# Diabetes Dataset (scikit-learn built-in)
#
# - Samples: 442
# - Features: 10 (all numerical, already standardized)
# - Original task: Regression
# - Converted task: Binary Classification
#
# Target Construction:
# - 0 → Low disease progression
# - 1 → High disease progression
# - Threshold used: Median of the target variable
#   (ensures balanced class distribution)
#
# ------------------------------------------------------------
# Problem Statement
# -----------------
# Predict whether a patient falls into a high-risk or low-risk
# diabetes category using tree-based classification models.
#
# ------------------------------------------------------------
# Models Used
# ----------
# 1) Decision Tree
#    - Baseline model
#    - Demonstrated overfitting when unrestricted
#    - Regularized using:
#        * max_depth
#        * min_samples_leaf
#
# 2) Random Forest
#    - Ensemble of decision trees
#    - Reduced variance compared to a single tree
#    - Improved ROC–AUC and stability
#
# 3) Extra Trees (Extremely Randomized Trees)
#    - Increased randomness in split selection
#    - Strong performance on tabular data
#    - Chosen as the primary model for tuning
#
# ------------------------------------------------------------
# Evaluation Metric
# -----------------
# Because this is a medical risk prediction task:
#
# - Primary metric: ROC–AUC
# - Secondary metrics: Accuracy, Confusion Matrix
#
# ROC–AUC was prioritized over accuracy to better evaluate
# ranking quality across decision thresholds.
#
# ------------------------------------------------------------
# Hyperparameter Tuning — GridSearchCV
# -----------------------------------
# GridSearchCV was used to obtain a more honest and
# generalizable estimate of model performance.
#
# GridSearch Setup:
# - Cross-validation: 5-fold
# - Scoring metric: roc_auc
#
# Parameter Grid (Extra Trees):
#
# param_grid = {
#     "n_estimators": [100, 200],
#     "max_depth": [3, 4, 5, None],
#     "min_samples_leaf": [1, 5, 10],
#     "max_features": ["sqrt", "log2"]
# }
#
# ------------------------------------------------------------
# Best Model (After GridSearchCV)
# ------------------------------
# Model: ExtraTreesClassifier
#
# Best Hyperparameters:
# - max_depth: None
# - min_samples_leaf: 5
# - n_estimators: 200
# - max_features: "sqrt"
#
# Best Cross-Validated ROC–AUC:
# - ~0.83
#
# ------------------------------------------------------------
# Key Insights
# -----------
# - Overfitting control via `min_samples_leaf` mattered more than tree depth.
# - Ensemble methods significantly outperformed single decision trees.
# - Cross-validated performance estimates are more reliable than a single
#   train–test split.
# - Increased randomness in Extra Trees improved generalization on tabular
#   medical data.
#
# ------------------------------------------------------------
# Project Conclusion
# ------------------
# This project demonstrates that:
# - Tree-based models effectively capture non-linear decision boundaries.
# - Ensemble learning improves stability and predictive performance.
# - Hyperparameter tuning using GridSearchCV leads to more honest and
#   generalizable model evaluation.
#
# The final tuned Extra Trees model provides a strong, well-regularized
# baseline for medical risk classification.
#
# ------------------------------------------------------------
# Tools & Libraries
# -----------------
# - Python
# - NumPy, Pandas
# - Scikit-learn
#
# ------------------------------------------------------------
# Author
# ------
# Tanveer Singh
# (Tree-Based ML Projects Series)
# ============================================================
