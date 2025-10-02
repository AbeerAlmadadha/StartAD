# Clean training using only original high-quality data
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

print("=== CLEAN TRAINING WITH ORIGINAL DATA ONLY ===")

# Load ONLY original data to avoid augmentation noise
df = pd.read_csv("data/teacher_notes_labeled.csv")
print(f"Using original dataset: {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Verify original data quality
print("\n=== VERIFYING ORIGINAL DATA QUALITY ===")
positive_samples = df[df['teacher_note'].str.contains('متفوقة|مشاركة|متميز|ممتاز', na=False)]
print("Positive samples in original data:")
for _, row in positive_samples.iterrows():
    print(f"  {row['teacher_note']} -> {row['label']}")

# Enhanced Arabic preprocessing
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text)

# Split with smaller test size due to limited data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Conservative TF-IDF settings for small dataset
print("\n=== CREATING CONSERVATIVE TF-IDF FEATURES ===")
vectorizer = TfidfVectorizer(
    max_features=500,  # Much smaller for limited data
    ngram_range=(1, 2),  # Unigrams and bigrams only
    analyzer='word',
    min_df=1,  # Allow single occurrences due to small dataset
    max_df=0.8,  # Remove very common words
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Conservative classifiers to avoid overfitting
classifiers = {
    'Naive Bayes (alpha=2.0)': MultinomialNB(alpha=2.0),  # High smoothing
    'Logistic Regression (C=0.1)': LogisticRegression(C=0.1, random_state=42, max_iter=1000),  # High regularization
}

best_accuracy = 0
best_classifier = None
best_name = None

print("\n=== TRAINING CONSERVATIVE CLASSIFIERS ===")
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

print(f"\n🏆 Best classifier: {best_name} with accuracy: {best_accuracy:.3f}")

# Save the clean model
print("\nSaving clean model...")
with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Clean model saved successfully!")

# Test the critical cases
print("\n=== TESTING CRITICAL CASES ===")
critical_test_cases = [
    ("الطالبة متفوقة ومشاركة", "LOW"),  # Should be LOW - positive performance
    ("الطالبة مشاركة ومتفوقة", "LOW"),  # Should be LOW - positive performance
    ("الطالب متميز في الدراسة", "LOW"),  # Should be LOW - positive performance
    ("الطالب غاب كثيراً ولم يسلم الواجبات", "HIGH"),  # Should be HIGH - serious issues
    ("الطالبة تعاني من صعوبات كبيرة", "HIGH"),  # Should be HIGH - serious issues
    ("الطالب يحتاج مساعدة بسيطة", "MODERATE"),  # Should be MODERATE - minor support needed
    ("الطالبة تتعرض لنوبات من القلق تمنعها من المشاركة", "HIGH"),  # Should be HIGH - anxiety prevents participation
]

correct_predictions = 0
total_predictions = len(critical_test_cases)

for note, expected in critical_test_cases:
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = best_classifier.predict(note_tfidf)[0]
    probabilities = best_classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(best_classifier.classes_, probabilities))

    is_correct = prediction.upper() == expected
    if is_correct:
        correct_predictions += 1
        status = "✅ CORRECT"
    else:
        status = f"❌ WRONG (expected {expected})"

    print(f"\nNote: {note}")
    print(f"Expected: {expected} | Predicted: {prediction.upper()} | {status}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probabilities - High: {class_probs.get('high', 0):.3f}, "
          f"Moderate: {class_probs.get('moderate', 0):.3f}, "
          f"Low: {class_probs.get('low', 0):.3f}")

accuracy_on_critical = correct_predictions / total_predictions
print(f"\n=== CRITICAL CASES ACCURACY: {accuracy_on_critical:.1%} ({correct_predictions}/{total_predictions}) ===")

if accuracy_on_critical < 0.8:
    print("⚠️  WARNING: Model still has issues with critical cases!")
    print("Consider collecting more training data or manual feature engineering.")
else:
    print("✅ Model performs well on critical cases!")

print("\n" + "="*60)
print("CLEAN TRAINING COMPLETED!")
print(f"Model trained on {len(df)} original samples only")
print(f"Test accuracy: {best_accuracy:.3f}")
print(f"Critical cases accuracy: {accuracy_on_critical:.1%}")
print("="*60)
