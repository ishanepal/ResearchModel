# Sample from Chat
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load data (assume data is preprocessed and formatted)
# X, y = load_data()  # Replace with actual data loading function

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

# Sample from deep
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Load data
# def load_data():
#     data = pd.read_csv("data.csv")  # Replace with your data file
#     X = data.drop("target", axis=1)  # Replace "target" with your target column
#     y = data["target"]
#     return X, y

# # Load data
# X, y = load_data()

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Train model
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# # Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Define features (X) and target (y)
X = df.drop(columns=["security_risk"])
y = df["security_risk"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
