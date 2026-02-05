# import numpy as np
# import os
# import joblib

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.utils import resample

# from dataset_loader import load_dataset


# def balance_data(X, y):
#     X0 = X[y == 0]  # Human
#     X1 = X[y == 1]  # AI

#     if len(X0) > len(X1):
#         X0 = resample(X0, n_samples=len(X1), random_state=42)
#     else:
#         X1 = resample(X1, n_samples=len(X0), random_state=42)

#     X_balanced = np.vstack((X0, X1))
#     y_balanced = np.array([0]*len(X0) + [1]*len(X1))

#     return X_balanced, y_balanced


# def main():
#     print("Loading dataset...")
#     X, y = load_dataset()

#     print(f"Total samples before balancing: {len(X)}")

#     if len(X) == 0:
#         print("No data found. Check dataset folder.")
#         return

#     X, y = balance_data(X, y)

#     print(f"Samples after balancing: {len(X)}")

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     model = RandomForestClassifier(
#         n_estimators=300,
#         max_depth=20,
#         class_weight="balanced",
#         random_state=42
#     )

#     print("Training model...")
#     model.fit(X_train, y_train)

#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)

#     print(f"Accuracy: {accuracy:.3f}")

#     # Save model
#     model_dir = os.path.join("..", "model")
#     os.makedirs(model_dir, exist_ok=True)

#     model_path = os.path.join(model_dir, "classifier.pkl")
#     joblib.dump(model, model_path)

#     print(f"Model saved at: {model_path}")


# if __name__ == "__main__":
#     main()
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from dataset_loader import load_dataset


def balance_data(X, y):
    X0 = X[y == 0]
    X1 = X[y == 1]

    n = min(len(X0), len(X1))
    X0 = resample(X0, n_samples=n, random_state=42)
    X1 = resample(X1, n_samples=n, random_state=42)

    X_balanced = np.vstack((X0, X1))
    y_balanced = np.array([0]*n + [1]*n)

    return X_balanced, y_balanced


def main():
    print("Loading dataset...")
    X, y = load_dataset()

    print(f"Total samples before balancing: {len(X)}")

    if len(X) == 0:
        print("No data found. Check dataset folder.")
        return

    X, y = balance_data(X, y)
    print(f"Samples after balancing: {len(X)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        class_weight="balanced",
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy:.3f}")

    model_dir = os.path.join("..", "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "classifier.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved at: {model_path}")


if __name__ == "__main__":
    main()
