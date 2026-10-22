from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def main():
    # (1) Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # (2) Split 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # (3) Baseline: One hidden layer with 3 neurons
    mlp_base = MLPClassifier(hidden_layer_sizes=(3,), max_iter=500, random_state=42)
    mlp_base.fit(X_train, y_train)
    
    base_train_acc = accuracy_score(y_train, mlp_base.predict(X_train)) * 100
    base_test_acc = accuracy_score(y_test, mlp_base.predict(X_test)) * 100
    
    print("--- Baseline Model (1 layer, 3 neurons) ---")
    print(f"Training Accuracy: {base_train_acc:.2f}%")
    print(f"Testing Accuracy: {base_test_acc:.2f}%\n")

    # (4) Adjusted Hyperparameters (e.g., activation, solver, learning rate)
    mlp_tuned = MLPClassifier(hidden_layer_sizes=(3,), activation='tanh', solver='lbfgs', max_iter=1000, random_state=42)
    mlp_tuned.fit(X_train, y_train)
    
    tuned_train_acc = accuracy_score(y_train, mlp_tuned.predict(X_train)) * 100
    tuned_test_acc = accuracy_score(y_test, mlp_tuned.predict(X_test)) * 100

    print("--- Tuned Hyperparameters Model (1 layer, 3 neurons) ---")
    print(f"Training Accuracy: {tuned_train_acc:.2f}%")
    print(f"Testing Accuracy: {tuned_test_acc:.2f}%")
    print("Explanation: Changing the solver to 'lbfgs' (better for small datasets) and activation to 'tanh' often helps the network converge to a better minima compared to default 'adam' and 'relu' given the severe bottleneck of only 3 neurons.\n")

    # (5) Increased Network Complexity
    mlp_complex = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp_complex.fit(X_train, y_train)
    
    complex_train_acc = accuracy_score(y_train, mlp_complex.predict(X_train)) * 100
    complex_test_acc = accuracy_score(y_test, mlp_complex.predict(X_test)) * 100

    print("--- Complex Model (2 layers, 10 neurons each) ---")
    print(f"Training Accuracy: {complex_train_acc:.2f}%")
    print(f"Testing Accuracy: {complex_test_acc:.2f}%")
    print("Explanation: Increasing the layers and neurons gives the network a higher capacity to capture the non-linear boundaries separating the Iris classes, leading to significantly improved classification performance.")

if __name__ == '__main__':
    main()
