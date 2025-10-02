# Debugging and improving the training data quality
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

print("=== DEBUGGING TRAINING DATA QUALITY ===")

# Load augmented data
df = pd.read_csv("data/teacher_notes_augmented.csv")
print(f"Total samples: {len(df)}")
print("Label distribution:")
print(df['label'].value_counts())

# Check for problematic patterns
print("\n=== ANALYZING PROBLEMATIC PATTERNS ===")

# Check samples with positive words but negative labels
positive_words = ['متفوقة', 'مشاركة', 'متميز', 'ممتاز', 'جيد', 'رائع']
negative_indicators = ['تمنعها', 'يمنعه', 'لا تستطيع', 'لا يستطيع', 'صعوبة', 'قلق', 'إحباط']

for word in positive_words:
    word_samples = df[df['teacher_note'].str.contains(word, na=False)]
    if len(word_samples) > 0:
        print(f"\nWord '{word}' distribution:")
        print(word_samples['label'].value_counts())

        # Check if positive word appears with negative context
        negative_context = word_samples[word_samples['teacher_note'].str.contains('|'.join(negative_indicators), na=False)]
        if len(negative_context) > 0:
            print(f"  ⚠️  '{word}' with negative context ({len(negative_context)} samples):")
            for _, row in negative_context.head(3).iterrows():
                print(f"    - {row['teacher_note']} -> {row['label']}")

# Enhanced preprocessing that considers context
def preprocess_arabic_text_enhanced(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

# Create enhanced features that consider negation and context
def create_enhanced_features(df):
    """Create features that better capture context and negation"""

    # Basic preprocessing
    df['clean_text'] = df['teacher_note'].apply(preprocess_arabic_text_enhanced)

    # Create negation-aware features
    df['has_negative_context'] = df['teacher_note'].str.contains(
        'تمنعها|يمنعه|لا تستطيع|لا يستطيع|لا ت|لا ي', na=False
    ).astype(int)

    df['has_positive_words'] = df['teacher_note'].str.contains(
        'متفوقة|متميز|ممتاز|جيد|رائع|مبدع', na=False
    ).astype(int)

    df['has_participation_words'] = df['teacher_note'].str.contains(
        'مشاركة|يشارك|تشارك|متفاعل', na=False
    ).astype(int)

    df['has_difficulty_words'] = df['teacher_note'].str.contains(
        'صعوبة|صعوبات|مشكلة|تحدي|عقبة', na=False
    ).astype(int)

    df['has_anxiety_words'] = df['teacher_note'].str.contains(
        'قلق|توتر|خوف|إحباط|يأس', na=False
    ).astype(int)

    # Combine features for better context understanding
    df['positive_without_negation'] = (df['has_positive_words'] == 1) & (df['has_negative_context'] == 0)
    df['participation_prevented'] = (df['has_participation_words'] == 1) & (df['has_negative_context'] == 1)

    return df

print("\n=== CREATING ENHANCED FEATURES ===")
df = create_enhanced_features(df)

# Split data
X = df[['clean_text']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Enhanced TF-IDF with better parameters for Arabic
print("\n=== CREATING ENHANCED TF-IDF FEATURES ===")
vectorizer = TfidfVectorizer(
    max_features=1500,  # Reduced to focus on most important features
    ngram_range=(1, 2),  # Reduced to bigrams to avoid overfitting
    analyzer='word',
    min_df=3,  # Increased to reduce noise
    max_df=0.9,  # More strict to remove very common words
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train['clean_text'])
X_test_tfidf = vectorizer.transform(X_test['clean_text'])

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Train and compare classifiers
classifiers = {
    'Naive Bayes (alpha=1.0)': MultinomialNB(alpha=1.0),  # More smoothing
    'Logistic Regression (C=0.5)': LogisticRegression(C=0.5, random_state=42, max_iter=1000),  # More regularization
    'Random Forest (balanced)': RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42)
}

best_accuracy = 0
best_classifier = None
best_name = None

print("\n=== TRAINING ENHANCED CLASSIFIERS ===")
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
print("\nSaving enhanced model...")
with open('saved_model/classifier.pkl', 'wb') as f:
    pickle.dump(best_classifier, f)

with open('saved_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Enhanced model saved successfully!")

# Test with problematic cases
print("\n=== TESTING PROBLEMATIC CASES ===")
test_cases = [
    "الطالبة متفوقة ومشاركة",  # Should be LOW
    "الطالب غاب 3 مرات ولم يسلم الواجب",  # Should be HIGH
    "الطالبة تتعرض لنوبات من القلق تمنعها من المشاركة",  # Should be HIGH (correctly)
    "الطالب يحتاج مساعدة إضافية في الرياضيات",  # Should be MODERATE
    "الطالبة متميزة في جميع المواد",  # Should be LOW
    "الطالب يعاني من صعوبات كبيرة",  # Should be HIGH
    "الطالبة مشاركة وتحب التعلم"  # Should be LOW
]

for note in test_cases:
    clean_note = preprocess_arabic_text_enhanced(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = best_classifier.predict(note_tfidf)[0]
    probabilities = best_classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(best_classifier.classes_, probabilities))

    print(f"\nNote: {note}")
    print(f"Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
    print(f"Probabilities - High: {class_probs.get('high', 0):.3f}, "
          f"Moderate: {class_probs.get('moderate', 0):.3f}, "
          f"Low: {class_probs.get('low', 0):.3f}")

print("\n" + "="*60)
print("ENHANCED TRAINING COMPLETED!")
print("The model should now better handle context and negation.")
print("="*60)
