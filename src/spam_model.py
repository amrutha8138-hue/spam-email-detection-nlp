import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv("../data/spam.csv")

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])

y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Output
print("Spam Detection Accuracy:", accuracy_score(y_test, predictions))
