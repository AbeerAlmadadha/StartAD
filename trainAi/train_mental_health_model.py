# Mental Health Risk Assessment Training Script
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

print("=== MENTAL HEALTH RISK ASSESSMENT MODEL TRAINING ===")

# Load mental health focused data
df = pd.read_csv("data/teacher_notes_mental_health.csv")
print(f"Using mental health dataset: {len(df)} samples")
print("Label distribution:")
print(df['label'].value_counts())

# Handle empty notes (they should be 'none')
df['teacher_note'] = df['teacher_note'].fillna('')
df.loc[df['teacher_note'].str.strip() == '', 'label'] = 'none'

print("\nAfter handling empty notes:")
print(df['label'].value_counts())

# Enhanced Arabic text preprocessing
def preprocess_arabic_text(text):
    if pd.isna(text) or text.strip() == "":
        return ""

    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.25, random_state=42, stratify=df['label']
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Training distribution:")
print(y_train.value_counts())

# Enhanced TF-IDF vectorizer for mental health text
print("\nCreating TF-IDF features for mental health assessment...")
vectorizer = TfidfVectorizer(
    max_features=1000,  # Focused feature set
    ngram_range=(1, 2),  # Unigrams and bigrams
    analyzer='word',
    min_df=1,  # Allow single occurrences due to limited data
    max_df=0.95,  # Remove very common words
    sublinear_tf=True
)

# Handle empty texts for vectorization
X_train_clean = [text if text else "عادي" for text in X_train]
X_test_clean = [text if text else "عادي" for text in X_test]

X_train_tfidf = vectorizer.fit_transform(X_train_clean)
X_test_tfidf = vectorizer.transform(X_test_clean)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train multiple classifiers optimized for mental health assessment
classifiers = {
    'Naive Bayes (alpha=1.0)': MultinomialNB(alpha=1.0),
    'Logistic Regression (balanced)': LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ),
    'Random Forest (balanced)': RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
}

best_accuracy = 0
best_classifier = None
best_name = None

print("\n=== TRAINING MENTAL HEALTH RISK CLASSIFIERS ===")
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

# Save the best model
print("\nSaving mental health risk assessment model...")
with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Mental health model saved successfully!")

# Test with mental health scenarios
print("\n=== TESTING MENTAL HEALTH RISK SCENARIOS ===")
test_scenarios = [
    ("", "none"),  # Empty note
    ("الطالب يظهر دافعية جيدة", "none"),  # Positive note
    ("الطالب يبدو متعباً أحياناً", "low"),  # Minor concern
    ("الطالبة تتردد في المشاركة وتبدو قلقة", "moderate"),  # Moderate concern
    ("الطالب يعاني من نوبات بكاء ويرفض التفاعل", "high"),  # High risk
    ("الطالبة تعاني من قلق شديد ولا تأتي للمدرسة", "high"),  # High risk
    ("الطالب يحتاج مساعدة في الرياضيات", "low"),  # Academic only
    ("الطالبة تظهر سلوكيات عدوانية مفاجئة", "high"),  # Behavioral red flag
]

correct_predictions = 0
total_predictions = len(test_scenarios)

for note, expected in test_scenarios:
    clean_note = preprocess_arabic_text(note) if note else "عادي"
    note_tfidf = vectorizer.transform([clean_note])
    prediction = best_classifier.predict(note_tfidf)[0]
    probabilities = best_classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(best_classifier.classes_, probabilities))

    is_correct = prediction == expected
    if is_correct:
        correct_predictions += 1
        status = "✅ CORRECT"
    else:
        status = f"❌ WRONG (expected {expected})"

    print(f"\nNote: '{note}' (Expected: {expected})")
    print(f"Predicted: {prediction} | {status}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Probabilities: None:{class_probs.get('none', 0):.3f} "
          f"Low:{class_probs.get('low', 0):.3f} "
          f"Mod:{class_probs.get('moderate', 0):.3f} "
          f"High:{class_probs.get('high', 0):.3f}")

scenario_accuracy = correct_predictions / total_predictions
print(f"\n=== MENTAL HEALTH SCENARIOS ACCURACY: {scenario_accuracy:.1%} ({correct_predictions}/{total_predictions}) ===")

print("\n" + "="*60)
print("MENTAL HEALTH RISK ASSESSMENT MODEL TRAINING COMPLETED!")
print(f"Model trained on {len(df)} samples")
print(f"Test accuracy: {best_accuracy:.3f}")
print(f"Scenario accuracy: {scenario_accuracy:.1%}")
print("="*60)

print("\n📋 Label Definitions:")
print("- NONE: No mental health concerns (empty notes or positive feedback)")
print("- LOW: Minor behavioral concerns that need monitoring")
print("- MODERATE: Concerning patterns requiring intervention")
print("- HIGH: Serious mental health risks requiring immediate action")
