# ===== Step 1: Imports =====
import pickle
import re
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===== Step 2: Load model =====
print("Loading model...")
with open('saved_model/classifier.pkl', 'rb') as f:
    model = pickle.load(f)

with open('saved_model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")

# ===== Step 3: Preprocessing function =====
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

# ===== Step 4: New notes =====
new_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة",
    "الطالب يحتاج مساعدة إضافية في الرياضيات",
    "الطالبة تواجه صعوبات كبيرة في الفهم",
    "الطالب متميز ويشارك بفعالية"
]

# ===== Step 5: Predict =====
print("\nPrediction Results:")
print("=" * 50)

for i, note in enumerate(new_notes, 1):
    # Preprocess the note
    clean_note = preprocess_arabic_text(note)

    # Vectorize the note
    note_tfidf = vectorizer.transform([clean_note])

    # Make prediction
    prediction = model.predict(note_tfidf)[0]
    probabilities = model.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(model.classes_, probabilities))

    # ===== Step 6: Show results =====
    print(f"\n{i}. Note: {note}")
    print(f"   Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
    print(f"   Probabilities:")
    for label in ['high', 'moderate', 'low']:
        prob = class_probs.get(label, 0)
        print(f"     - {label}: {prob:.3f}")

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n" + "=" * 50)
print("Legend:")
print("- LOW: Student is performing well, no significant concerns")
print("- MODERATE: Student needs some support or has minor issues")
print("- HIGH: Student is struggling and needs immediate attention")
print("=" * 50)
