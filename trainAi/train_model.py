# ===== Step 1: Imports =====
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re

# ===== Step 2: Load data =====
# Use augmented data if available, otherwise use original
try:
    df = pd.read_csv("data/teacher_notes_augmented.csv")
    print(f"Using augmented dataset: {len(df)} samples")
except FileNotFoundError:
    df = pd.read_csv("data/teacher_notes_labeled.csv")
    print(f"Using original dataset: {len(df)} samples")

print("Label distribution:")
print(df['label'].value_counts())

# ===== Step 3: Preprocess text =====
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text)

# ===== Step 4: Split data =====
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# ===== Step 5: Vectorize =====
print("\nCreating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3),
    analyzer='word',
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# ===== Step 6: Train models =====
classifiers = {
    'Naive Bayes': MultinomialNB(alpha=0.5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_accuracy = 0
best_classifier = None
best_name = None

print("\nTraining multiple classifiers...")
for name, classifier in classifiers.items():
    print(f"\n--- {name} ---")

    classifier.fit(X_train_tfidf, y_train)
    y_pred = classifier.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = classifier
        best_name = name

# ===== Step 7: Save best model =====
print(f"\n🏆 Best classifier: {best_name} with accuracy: {best_accuracy:.3f}")
print("\nSaving model...")

with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Model saved successfully!")

# ===== Step 8: Test predictions =====
test_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة"
]

print("\nTesting saved model:")
for note in test_notes:
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = best_classifier.predict(note_tfidf)[0]
    probabilities = best_classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    print(f"Note: {note}")
    print(f"Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
    print()

print("Training completed successfully!")
