# Lightweight classification for Arabic teacher notes
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re

print("Loading data...")
# Load data
df = pd.read_csv("data/teacher_notes_labeled.csv")
print(f"Loaded {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Simple Arabic text preprocessing
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Create TF-IDF vectorizer for Arabic text
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),  # unigrams and bigrams
    analyzer='word'
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train Naive Bayes classifier
print("\nTraining classifier...")
classifier = MultinomialNB(alpha=1.0)
classifier.fit(X_train_tfidf, y_train)

# Make predictions
print("Making predictions...")
y_pred = classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.3f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
print("\nSaving model...")
with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")

# Test with sample predictions
print("\nTesting with sample notes:")
test_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة",
    "الطالب يحتاج مساعدة إضافية في الرياضيات",
    "الطالبة تواجه صعوبات كبيرة في الفهم",
    "الطالب متميز ويشارك بفعالية"
]

for note in test_notes:
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = classifier.predict(note_tfidf)[0]
    probabilities = classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    print(f"Note: {note}")
    print(f"Prediction: {prediction} (confidence: {confidence:.3f})")
    print(f"Probabilities - Low: {probabilities[2]:.3f}, Moderate: {probabilities[1]:.3f}, High: {probabilities[0]:.3f}")
    print()

print("Testing completed successfully!")
