import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Example dataset (replace with a larger CSV for real use)
data = {
    "text": [
        "Win money now!",
        "Meeting at 10am",
        "Click here for prize",
        "Project deadline tomorrow"
    ],
    "label": ["phishing", "safe", "phishing", "safe"]
}
df = pd.DataFrame(data)

# Features and labels
X = df["text"]
y = df["label"]

# Convert text to numeric features
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.3, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
