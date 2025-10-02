# Comprehensive model testing script
import pickle
import re
import pandas as pd

def preprocess_arabic_text(text):
    """Preprocess Arabic text consistently with training"""
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep only Arabic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Remove common Arabic diacritics (tashkeel) if any
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)

    return text

def load_model():
    """Load the trained model and vectorizer"""
    print("Loading model...")
    with open('saved_model/classifier.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('saved_model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    print(f"Model loaded: {type(model).__name__}")
    print(f"Vectorizer loaded: {type(vectorizer).__name__}")
    return model, vectorizer

def predict_note(note, model, vectorizer):
    """Make prediction for a single note"""
    clean_note = preprocess_arabic_text(note)
    note_tfidf = vectorizer.transform([clean_note])
    prediction = model.predict(note_tfidf)[0]
    probabilities = model.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    # Get probabilities for each class
    class_probs = dict(zip(model.classes_, probabilities))

    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': class_probs
    }

def run_comprehensive_tests():
    """Run comprehensive tests across all categories"""

    # Load model
    model, vectorizer = load_model()

    # Comprehensive test cases
    test_cases = {
        "LOW Priority (Excellent Performance)": [
            ("الطالبة متفوقة ومشاركة في جميع الأنشطة", "low"),
            ("الطالب متميز في الدراسة ويحب التعلم", "low"),
            ("الطالبة تتفوق أكاديمياً وتساعد زملاءها", "low"),
            ("الطالب منظم ومتفاعل مع المعلمين", "low"),
            ("الطالبة لديها مهارات قيادية رائعة", "low"),
            ("الطالب يشارك بحماس في النقاشات", "low"),
            ("الطالبة تبدي فهماً عميقاً للمواد", "low"),
            ("الطالب ملتزم بالحضور والواجبات", "low"),
            ("الطالبة تظهر إبداعاً في حل المسائل", "low"),
            ("الطالب متعاون ومحبوب من زملائه", "low")
        ],

        "MODERATE Priority (Needs Support)": [
            ("الطالبة تحتاج مساعدة إضافية في الرياضيات", "moderate"),
            ("الطالب يبذل جهداً لكن يحتاج تشجيعاً", "moderate"),
            ("الطالبة تتردد في الإجابة أحياناً", "moderate"),
            ("الطالب يحتاج وقتاً أطول لفهم الدروس", "moderate"),
            ("الطالبة تحتاج تطوير مهارات الكتابة", "moderate"),
            ("الطالب أحياناً يتشتت لكنه مجتهد", "moderate"),
            ("الطالبة تحتاج دعماً في الثقة بالنفس", "moderate"),
            ("الطالب يحتاج متابعة في أداء الواجبات", "moderate"),
            ("الطالبة تحتاج تحسين مهارات التنظيم", "moderate"),
            ("الطالب يحتاج مساعدة في إدارة الوقت", "moderate")
        ],

        "HIGH Priority (Serious Concerns)": [
            ("الطالبة تعاني من صعوبات كبيرة في التعلم", "high"),
            ("الطالب غاب كثيراً ولم يسلم الواجبات", "high"),
            ("الطالبة تواجه مشاكل سلوكية خطيرة", "high"),
            ("الطالب يعاني من قلق شديد يؤثر على دراسته", "high"),
            ("الطالبة لا تستطيع التركيز نهائياً", "high"),
            ("الطالب يعاني من إحباط شديد ويرفض المشاركة", "high"),
            ("الطالبة تتعرض لنوبات قلق تمنعها من المشاركة", "high"),
            ("الطالب لديه صعوبات جسيمة في القراءة والكتابة", "high"),
            ("الطالبة منعزلة تماماً عن زملائها", "high"),
            ("الطالب يظهر علامات اكتئاب واضحة", "high")
        ],

        "Edge Cases (Tricky Examples)": [
            ("الطالبة مشاركة لكن تحتاج تشجيعاً", "moderate"),
            ("الطالب متفوق لكن خجول جداً", "moderate"),
            ("الطالبة تحب المادة لكن تجد صعوبة فيها", "moderate"),
            ("الطالب ذكي لكن كسول في أداء الواجبات", "moderate"),
            ("الطالبة متفاعلة لكن تنسى الواجبات أحياناً", "moderate"),
            ("الطالب موهوب لكن لا يثق بنفسه", "moderate"),
            ("الطالبة نشطة لكن تتشتت بسرعة", "moderate"),
            ("الطالب يشارك لكن يحتاج تنظيماً أكثر", "moderate")
        ]
    }

    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("="*80)

    total_correct = 0
    total_tests = 0
    category_results = {}

    for category, cases in test_cases.items():
        print(f"\n📋 {category}")
        print("-" * 60)

        correct_in_category = 0

        for i, (note, expected) in enumerate(cases, 1):
            result = predict_note(note, model, vectorizer)
            predicted = result['prediction']
            confidence = result['confidence']
            probs = result['probabilities']

            is_correct = predicted == expected
            if is_correct:
                correct_in_category += 1
                total_correct += 1
                status = "✅"
            else:
                status = "❌"

            total_tests += 1

            print(f"{status} {i:2d}. {note}")
            print(f"     Expected: {expected.upper():8} | Predicted: {predicted.upper():8} | Confidence: {confidence:.3f}")
            print(f"     Probabilities: H:{probs.get('high', 0):.3f} M:{probs.get('moderate', 0):.3f} L:{probs.get('low', 0):.3f}")
            print()

        category_accuracy = correct_in_category / len(cases)
        category_results[category] = {
            'correct': correct_in_category,
            'total': len(cases),
            'accuracy': category_accuracy
        }

        print(f"📊 Category Accuracy: {category_accuracy:.1%} ({correct_in_category}/{len(cases)})")

    # Overall results
    overall_accuracy = total_correct / total_tests

    print("\n" + "="*80)
    print("📈 OVERALL TEST RESULTS")
    print("="*80)

    for category, results in category_results.items():
        acc = results['accuracy']
        color = "🟢" if acc >= 0.8 else "🟡" if acc >= 0.6 else "🔴"
        print(f"{color} {category:<35}: {acc:>6.1%} ({results['correct']}/{results['total']})")

    print(f"\n🎯 OVERALL ACCURACY: {overall_accuracy:.1%} ({total_correct}/{total_tests})")

    # Performance analysis
    print("\n" + "="*80)
    print("🔍 PERFORMANCE ANALYSIS")
    print("="*80)

    if overall_accuracy >= 0.85:
        print("🟢 EXCELLENT: Model performs very well across all categories!")
    elif overall_accuracy >= 0.75:
        print("🟡 GOOD: Model performs well but has room for improvement.")
    elif overall_accuracy >= 0.65:
        print("🟠 FAIR: Model has decent performance but needs refinement.")
    else:
        print("🔴 POOR: Model needs significant improvement.")

    # Specific recommendations
    print("\n📝 RECOMMENDATIONS:")

    low_acc = category_results["LOW Priority (Excellent Performance)"]['accuracy']
    mod_acc = category_results["MODERATE Priority (Needs Support)"]['accuracy']
    high_acc = category_results["HIGH Priority (Serious Concerns)"]['accuracy']
    edge_acc = category_results["Edge Cases (Tricky Examples)"]['accuracy']

    if low_acc < 0.8:
        print("- ⚠️  Improve recognition of excellent student performance")
    if mod_acc < 0.8:
        print("- ⚠️  Better distinguish moderate support needs")
    if high_acc < 0.8:
        print("- ⚠️  More training needed for serious concern detection")
    if edge_acc < 0.7:
        print("- ⚠️  Edge cases need more nuanced training data")

    if overall_accuracy >= 0.8:
        print("- ✅ Model is ready for production use!")
    else:
        print("- 📚 Consider adding more training data")
        print("- 🔧 Fine-tune model parameters")
        print("- 📖 Review misclassified cases for patterns")

    print("\n" + "="*80)
    print("🏁 TESTING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    run_comprehensive_tests()
