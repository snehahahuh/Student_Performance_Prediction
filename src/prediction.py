import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Data preprocessing
data = data.dropna()

# Input features and target
X = data[["Hours_Studied", "Attendance", "Previous_Score", "Assignments"]]
y = data["Final_Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Student Performance Prediction")
print("--------------------------------")
print("Model: Decision Tree")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Sample student prediction
sample = pd.DataFrame(
    [[5, 82, 63, 7]],
    columns=["Hours_Studied", "Attendance", "Previous_Score", "Assignments"]
)

prediction = model.predict(sample)

print("Sample Student Prediction:", prediction[0])