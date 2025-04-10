import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os
import nltk

nltk.download('stopwords')

# Dataset folder paths (update these paths as needed)
TRAIN_PATH = "./train.csv"
TEST_PATH = "./test.csv"

# Load datasets (semicolon-separated)
df_train = pd.read_csv(TRAIN_PATH, sep=';')
df_test = pd.read_csv(TEST_PATH, sep=';')

# Drop any rows with missing values
df_train.dropna(inplace=True)
df_test.dropna(inplace=True)

# Extract features and labels
X_train = df_train['text']
y_train = df_train['label']
X_test = df_test['text']
y_test = df_test['label']

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Prediction & Evaluation
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {acc * 100:.2f}%")
print("🧾 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model and vectorizer
output_dir = "/opt/ml/output"
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open(f"{output_dir}/vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print(f"📦 Model saved to {output_dir}/model.pkl")
print(f"📦 Vectorizer saved to {output_dir}/vectorizer.pkl")
