# Prediction script for lightweight Arabic teacher notes classifier
import pickle
import re

# Load the saved model and vectorizer
print("Loading model...")
with open('saved_model/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('saved_model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model loaded successfully!")

# Preprocessing function
def preprocess_arabic_text(text):
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Prediction function
def predict_note(note):
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = classifier.predict(note_tfidf)[0]
    probabilities = classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Map probabilities to correct labels
    label_mapping = classifier.classes_
    prob_dict = {}
    for i, label in enumerate(label_mapping):
        prob_dict[label] = probabilities[i]

    return prediction, confidence, prob_dict

# Test with new notes
test_notes = [
    "الطالب غاب 3 مرات ولم يسلم الواجب",
    "الطالبة مشاركة ومتفوقة",
    "الطالب يحتاج مساعدة إضافية في الرياضيات",
    "الطالبة تواجه صعوبات كبيرة في الفهم",
    "الطالب متميز ويشارك بفعالية",
    "الطالبة لا تركز في الصف وتحتاج متابعة مستمرة",
    "الطالب يحل واجباته دائماً ويتفاعل مع الدروس"
]

print("\n" + "="*60)
print("TEACHER NOTES CLASSIFICATION RESULTS")
print("="*60)

for i, note in enumerate(test_notes, 1):
    prediction, confidence, probabilities = predict_note(note)

    print(f"\n{i}. Note: {note}")
    print(f"   Prediction: {prediction.upper()} (confidence: {confidence:.3f})")
    print(f"   Probabilities:")
    for label, prob in probabilities.items():
        print(f"     - {label}: {prob:.3f}")

print("\n" + "="*60)
print("Legend:")
print("- LOW: Student is performing well, no significant concerns")
print("- MODERATE: Student needs some support or has minor issues")
print("- HIGH: Student is struggling and needs immediate attention")
print("="*60)

# Interactive mode
print("\nInteractive Mode - Enter your own teacher notes!")
print("Type 'quit' to exit")

while True:
    user_note = input("\nEnter a teacher note in Arabic: ").strip()

    if user_note.lower() == 'quit':
        print("Goodbye!")
        break

    if not user_note:
        print("Please enter a valid note.")
        continue

    try:
        prediction, confidence, probabilities = predict_note(user_note)
        print(f"\nPrediction: {prediction.upper()} (confidence: {confidence:.3f})")
        print("Probabilities:")
        for label, prob in probabilities.items():
            print(f"  - {label}: {prob:.3f}")
    except Exception as e:
        print(f"Error processing note: {e}")
