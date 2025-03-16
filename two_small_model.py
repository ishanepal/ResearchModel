import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set pandas to display all columns
pd.set_option('display.max_columns', None)

# Load data
def load_data():
    data = pd.read_csv("data1.csv")  # Load the CSV file
    print("Dataset Head:")
    print(data.head())  # Debug: Print the first few rows of the dataset
    X = data.drop("target", axis=1)  # Features (all columns except 'target')
    y = data["target"]  # Target column
    return X, y

# Preprocess data
def preprocess_data(X, y):
    # Convert categorical target labels to binary (0 = safe, 1 = risky)
    y = y.map({"safe": 0, "risky": 1})
    print("\nClass Distribution:")
    print(y.value_counts())  # Debug: Print the distribution of classes
    return X, y

# Train and evaluate model
def train_and_evaluate(X, y):
    # Split data into training and testing sets (with stratified sampling)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)
    
    # Debug: Print the shape of train and test sets
    print("\nTrain Set Shape:", X_train.shape)
    print("Test Set Shape:", X_test.shape)
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Main function
def main():
    # Load data
    X, y = load_data()

    # Preprocess data
    X, y = preprocess_data(X, y)

    # Train and evaluate model
    train_and_evaluate(X, y)

if __name__ == "__main__":
    main()