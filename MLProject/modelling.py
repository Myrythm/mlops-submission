import os
import sys
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# LOAD DATA
# =============================================================================
def load_data(train_path, test_path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Jika path relatif, gabungkan dengan script_dir agar aman saat dijalankan dari folder mana saja
    if not os.path.isabs(train_path):
        train_path = os.path.join(script_dir, train_path)
    if not os.path.isabs(test_path):
        test_path = os.path.join(script_dir, test_path)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop("Churn", axis=1)
    y_train = train_df["Churn"]
    X_test = test_df.drop("Churn", axis=1)
    y_test = test_df["Churn"]

    return X_train, X_test, y_train, y_test


# =============================================================================
# TRAINING DENGAN MLFLOW AUTOLOG
# =============================================================================
def train_model(n_estimators, max_depth, train_path, test_path):
    X_train, X_test, y_train, y_test = load_data(train_path, test_path)

    # Aktifkan autolog dari MLflow untuk Scikit-Learn
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Basic"):
        # Inisialisasi model dengan hyperparameter dari argument
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        # Training
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("=" * 50)
        print("HASIL TRAINING (Basic - Autolog)")
        print("=" * 50)
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print("=" * 50)
        print(f"MLflow Tracking URI : {mlflow.get_tracking_uri()}")
        print("=" * 50)

    # Matikan autolog setelah selesai
    mlflow.sklearn.autolog(disable=True)

    return model


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Nilai default jika dijalankan manual tanpa parameter
    n_estimators = 200
    max_depth = 10
    train_dataset = "data/telco_customer_churn_train.csv"
    test_dataset = "data/telco_customer_churn_test.csv"

    # Parsing arguments dari MLproject / command line
    if len(sys.argv) > 1:
        n_estimators = int(sys.argv[1])
    if len(sys.argv) > 2:
        max_depth = int(sys.argv[2])
    if len(sys.argv) > 3:
        train_dataset = sys.argv[3]
    if len(sys.argv) > 4:
        test_dataset = sys.argv[4]

    model = train_model(n_estimators, max_depth, train_dataset, test_dataset)
