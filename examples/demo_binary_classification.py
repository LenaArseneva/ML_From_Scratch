import numpy as np
from mlscratch.datasets import make_binary_classification
from mlscratch.model_selection import train_test_split
from mlscratch.logistic_regression import LogisticRegression
from mlscratch.metrics import accuracy, f1_score, roc_auc_score

def main():
    X, y = make_binary_classification(n_samples=800, n_features=4, random_state=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

    model = LogisticRegression(lr=0.2, n_iter=4000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Accuracy:", round(accuracy(y_test, y_pred), 4))
    print("F1:", round(f1_score(y_test, y_pred), 4))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))

if __name__ == "__main__":
    main()
