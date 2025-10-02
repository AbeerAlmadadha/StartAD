# Enhanced training script using augmented data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import re

print("Loading augmented training data...")
# Load augmented data
df = pd.read_csv("data/teacher_notes_augmented.csv")
print(f"Loaded {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Enhanced Arabic text preprocessing
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text)

# Split data with larger test set to get better evaluation
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training distribution:")
print(y_train.value_counts())

# Enhanced TF-IDF vectorizer for Arabic text
print("\nCreating enhanced TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=2000,  # Increased from 1000
    ngram_range=(1, 3),  # Include trigrams
    analyzer='word',
    min_df=2,  # Ignore terms that appear in less than 2 documents
    max_df=0.95,  # Ignore terms that appear in more than 95% of documents
    sublinear_tf=True  # Apply sublinear tf scaling
)

# Fit and transform training data
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Try multiple classifiers
classifiers = {
    'Naive Bayes': MultinomialNB(alpha=0.5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

best_accuracy = 0
best_classifier = None
best_name = None

print("\nTraining and comparing multiple classifiers...")
for name, classifier in classifiers.items():
    print(f"\n--- {name} ---")

    # Train classifier
    classifier.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test_tfidf)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.3f}")

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Keep track of best classifier
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_classifier = classifier
        best_name = name

print(f"\n🏆 Best classifier: {best_name} with accuracy: {best_accuracy:.3f}")

# Save the best model and vectorizer
print(f"\nSaving best model ({best_name})...")
with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Enhanced model saved successfully!")

# Enhanced testing with confidence scores
print("\nTesting with sample notes and confidence analysis:")
test_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة",
    "الطالب يحتاج مساعدة إضافية في الرياضيات",
    "الطالبة تواجه صعوبات كبيرة في الفهم",
    "الطالب متميز ويشارك بفعالية",
    "الطالبة متفوقة جداً في جميع المواد",
    "الطالب يعاني من قلق شديد",
    "الطالبة تحتاج دعماً بسيطاً"
]

for note in test_notes:
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = best_classifier.predict(note_tfidf)[0]
    probabilities = best_classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(best_classifier.classes_, probabilities))

    print(f"\nNote: {note}")
    print(f"Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
    print(f"Probabilities - High: {class_probs.get('high', 0):.3f}, "
          f"Low: {class_probs.get('low', 0):.3f}, "
          f"Moderate: {class_probs.get('moderate', 0):.3f}")

print("\n" + "="*60)
print("TRAINING COMPLETED WITH AUGMENTED DATA!")
print(f"Dataset expanded from 51 to {len(df)} samples")
print(f"Best model: {best_name}")
print(f"Best accuracy: {best_accuracy:.3f}")
print("="*60)
