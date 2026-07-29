import time

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score


TEST_SIZE = 0.3


def main():

    # fetch dataset
    car_evaluation = fetch_ucirepo(id=19)

    
    X = car_evaluation.data.features
    y = car_evaluation.data.targets

    # metadata
    print(car_evaluation.metadata)

    # variable information
    print(car_evaluation.variables)

    # encode features
    X_encoded = encode_features(X)

    # encode labels
    y_encoded = encode_labels(y)

    # basic exploration
    explore_data(X_encoded, y_encoded)

    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=TEST_SIZE, random_state=42
    )

    print("\nWITHOUT SCALING")
    

    # perceptron
    p_acc, p_train_time, p_predict_time = run_perceptron(
        x_train, x_test, y_train, y_test
    )

    print("Perceptron Accuracy:", round(p_acc, 4))
    print("Perceptron Training Time:", round(p_train_time, 6))
    print("Perceptron Prediction Time:", round(p_predict_time, 6))

    # knn
    k_values = [1, 3, 5]

    for k in k_values:
        acc, train_time, predict_time = run_knn(
            x_train, x_test, y_train, y_test, k
        )

        print("\nk =", k)
        print("Accuracy:", round(acc, 4))
        print("Training Time:", round(train_time, 6))
        print("Prediction Time:", round(predict_time, 6))

    print("\nWITH SCALING")
    

    x_train_scaled, x_test_scaled = scale_data(x_train, x_test)

    # perceptron with scaling
    p_acc_scaled, p_train_scaled, p_predict_scaled = run_perceptron(
        x_train_scaled, x_test_scaled, y_train, y_test
    )

    print("Perceptron Accuracy (scaled):", round(p_acc_scaled, 4))
    print("Perceptron Training Time (scaled):", round(p_train_scaled, 6))
    print("Perceptron Prediction Time (scaled):", round(p_predict_scaled, 6))

    # knn with scaling
    for k in k_values:
        acc, train_time, predict_time = run_knn(
            x_train_scaled, x_test_scaled, y_train, y_test, k
        )

        print("\nk =", k, "(scaled)")
        print("Accuracy:", round(acc, 4))
        print("Training Time:", round(train_time, 6))
        print("Prediction Time:", round(predict_time, 6))


def encode_features(X):
    X_encoded = X.copy()

    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])

    return X_encoded


def encode_labels(y):
    le = LabelEncoder()
    return le.fit_transform(y.values.ravel())


def explore_data(X, y):
    print("\nNumber of samples:", len(X))
    print("Number of features:", len(X.columns))

    counts = {}

    for label in y:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    print("\nClass distribution:")
    for label in counts:
        print("Class", label, ":", counts[label])


def scale_data(x_train, x_test):
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled


def run_perceptron(x_train, x_test, y_train, y_test):
    start_train = time.time()

    model = Perceptron(random_state=42)
    model.fit(x_train, y_train)

    end_train = time.time()

    start_predict = time.time()
    predictions = model.predict(x_test)
    end_predict = time.time()

    acc = accuracy_score(y_test, predictions)

    return acc, end_train - start_train, end_predict - start_predict


def run_knn(x_train, x_test, y_train, y_test, k):
    start_train = time.time()

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    end_train = time.time()

    start_predict = time.time()
    predictions = model.predict(x_test)
    end_predict = time.time()

    acc = accuracy_score(y_test, predictions)

    return acc, end_train - start_train, end_predict - start_predict


if __name__ == "__main__":
    main()