import os
from pathlib import Path
import re
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Optional heavy deps (lazy import when needed in __main__ path)
# from transformers import AutoTokenizer, AutoModel
# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns

def preprocess_arabic_text(text):
    # إزالة أي شيء ليس حرف عربي أو مسافة
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    ad = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = ad.sub('',text)
    return text

# -----------------------------
# Lightweight inference API
# -----------------------------

_ROOT = Path(__file__).parent
_MODEL_DIR = _ROOT / "trainAi" / "saved_model"

_VECTORIZER = None
_TXT_CLASSIFIER = None


def _load_text_classifier_if_available():
    global _VECTORIZER, _TXT_CLASSIFIER
    try:
        vec_path = _MODEL_DIR / "vectorizer.pkl"
        clf_path = _MODEL_DIR / "classifier.pkl"
        if vec_path.exists() and clf_path.exists():
            _VECTORIZER = joblib.load(vec_path)
            _TXT_CLASSIFIER = joblib.load(clf_path)
    except Exception:
        # Fail silently; we'll fall back to heuristics
        _VECTORIZER = None
        _TXT_CLASSIFIER = None


def predict_risk_level_from_note(note_text: str) -> str:
    """
    Predict a risk level label from a teacher note.
    Attempts to use a saved text classifier + vectorizer if available.
    Falls back to keyword heuristics when models are unavailable.

    Returns one of: ['low', 'medium', 'high']
    """
    if note_text is None:
        return "low"

    text = preprocess_arabic_text(str(note_text))

    # Try model first
    if _VECTORIZER is None or _TXT_CLASSIFIER is None:
        _load_text_classifier_if_available()

    if _VECTORIZER is not None and _TXT_CLASSIFIER is not None:
        try:
            X = _VECTORIZER.transform([text])
            pred = _TXT_CLASSIFIER.predict(X)[0]
            # Normalize possible outputs to low/medium/high
            normalized = str(pred).strip().lower()
            if "high" in normalized or "مرتفع" in normalized:
                return "high"
            if "medium" in normalized or "متوسط" in normalized:
                return "medium"
            return "low"
        except Exception:
            # fall through to heuristics
            pass

    # Heuristic fallback based on keyword presence
    high_keywords = [
        "عنف", "عنيف", "تنمر", "تهديد", "اكتئاب", "انتحار", "مخدرات", "اعتداء", "إيذاء",
        "self-harm", "bullying", "violence",
    ]
    medium_keywords = [
        "قلق", "توتر", "انعزال", "مشاكل أسرية", "مشاكل عائلية", "حزن", "تراجع", "عدوانية",
    ]

    txt = text
    if any(kw in txt for kw in high_keywords):
        return "high"
    if any(kw in txt for kw in medium_keywords):
        return "medium"
    return "low"


def suggest_actions_from_risk(risk_level: str) -> Dict[str, List[str]]:
    """
    Map risk level to suggested actions and supportive steps.
    Returns dict with keys: rationale, immediate_actions, follow_up
    """
    risk = (risk_level or "low").lower()
    if risk == "high":
        return {
            "rationale": [
                "تم رصد مؤشرات خطورة مرتفعة في الملاحظة.",
                "يُنصح بتدخل فوري واستشارة الاختصاصيين.",
            ],
            "immediate_actions": [
                "إخطار الأخصائي النفسي/الاجتماعي في المدرسة فورًا.",
                "التواصل مع ولي الأمر وفق سياسات المدرسة.",
                "تأمين بيئة آمنة للطالب ومتابعة حالته بشكل لصيق.",
            ],
            "follow_up": [
                "إحالة إلى جهة مختصة خارج المدرسة عند الحاجة (مثل NCMH).",
                "خطة دعم فردية وتقييم أسبوعي للتقدم.",
            ],
        }
    if risk == "medium":
        return {
            "rationale": [
                "مؤشرات متوسطة تتطلب متابعة وتقييم أقرب.",
            ],
            "immediate_actions": [
                "جلسة إرشادية أولية مع الأخصائي في المدرسة.",
                "تنظيم خطة متابعة قصيرة المدى وإشراك المعلمين المعنيين.",
            ],
            "follow_up": [
                "مواد توعوية ودعم اجتماعي داخل الصف.",
                "مراجعة أسبوعية للمؤشرات وتحديث الخطة عند الحاجة.",
            ],
        }
    # low
    return {
        "rationale": [
            "لا توجد مؤشرات خطورة واضحة في الملاحظة.",
        ],
        "immediate_actions": [
            "المتابعة الدورية والتشجيع الإيجابي داخل الصف.",
        ],
        "follow_up": [
            "مواد توعوية عامة وتعزيز مهارات التكيف.",
        ],
    }


def generate_ai_suggestions(student_id: str, teacher_note: str) -> Dict[str, object]:
    """
    Contract:
    - Input: student_id (str), teacher_note (str)
    - Output: dict with keys {student_id, risk_level, recommended_action, rationale, immediate_actions, follow_up}
    - Errors: function is defensive; returns 'low' when input is missing.
    """
    risk = predict_risk_level_from_note(teacher_note)
    mapping = {
        "high": "Immediate Intervention",
        "medium": "Teacher Monitoring + Awareness Materials",
        "low": "Regular Monitoring",
    }
    action = mapping.get(risk, "Regular Monitoring")
    details = suggest_actions_from_risk(risk)
    return {
        "student_id": str(student_id or "").strip(),
        "risk_level": risk,
        "recommended_action": action,
        **details,
    }


# -----------------------------
# Original training/data pipeline (disabled on import)
# -----------------------------
if __name__ == "__main__":
    # Heavy training/data code moved under __main__ guard so importing this module is safe.
    # If you want to retrain or run the full pipeline, ensure data files exist then run:
    #   python final.py

    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        AutoTokenizer = AutoModel = torch = None
        plt = sns = None

    # Safety: check paths before running
    notes_path = _ROOT / "data" / "teacher_notes_500.csv"
    synth_path = _ROOT / "data" / "synthetic_students_500_AI_.csv"

    if not notes_path.exists() or not synth_path.exists() or AutoTokenizer is None:
        print("Training/data files or transformers not available. Skipping heavy pipeline.")
    else:
        # 1- قراءة بيانات الملاحظات
        df_notes = pd.read_csv(notes_path)

        # 2- تحميل AraBERT
        tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
        model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
        model.eval()

        # 3- تحويل النصوص إلى embeddings
        df_notes['teacher_note'] = df_notes['teacher_note'].apply(preprocess_arabic_text)

        def get_embeddings(texts, tokenizer, model, max_len=128):
            inputs = tokenizer(
                texts.tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_len
            )
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.numpy()

        X_embeddings = get_embeddings(df_notes['teacher_note'], tokenizer, model)
        y = df_notes['label']

        # 4- تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X_embeddings, y, test_size=0.2, random_state=42, stratify=y
        )

        # 5- تدريب Classifier على embeddings
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=500)
        clf.fit(X_train, y_train)

        # 6- تقييم النموذج
        y_pred = clf.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

        # 7- التنبؤ لكل الطلاب وحفظ النتائج
        df = pd.read_csv(synth_path, dtype={"AI_RecommendedAction": "object"})
        df['RiskLevel'] = clf.predict(get_embeddings(df['teacher_note'], tokenizer, model))
        df.to_csv(_ROOT / "synthetic_students_500.csv", index=False)

        # Cleaning & feature engineering (kept as in original script)
        num_cols = ['Age', 'AttendanceRate', 'AcademicPerformance', 'CounselingSessions', 'HotlineCalls']
        df[num_cols] = df[num_cols].fillna(-1)
        text_cols = ['Name', 'BehaviorIncidents', 'RiskLevel']
        df[text_cols] = df[text_cols].fillna("unknown")
        df.drop_duplicates(subset=['StudentID'], inplace=True)
        df['GenderNum'] = df['Gender'].map({'M':0, 'F':1})
        df['BehaviorIncidentsNum'] = df['BehaviorIncidents'].factorize()[0]
        df['RiskLevel'] = df['RiskLevel'].factorize()[0]
        df['AttendanceRate'] = df['AttendanceRate'].clip(0, 100)
        df['AcademicPerformance'] = df['AcademicPerformance'].clip(0, 100)
        df['CounselingSessions'] = df['CounselingSessions'].clip(0)

        def assign_action(row):
            if (
                row['AttendanceRate'] < 65
                or row['AcademicPerformance'] < 57
                or row['BehaviorIncidents'] == "عنيف"
                or row['RiskLevel'] == 'high'
            ):
                return "Immediate Intervention"
            elif row['CounselingSessions'] > 2:
                return "Counseling Referral"
            elif row['AcademicPerformance'] > 88 or row['AttendanceRate'] > 90 or row['RiskLevel'] == 'low':
                return "Regular Monitoring"
            else:
                return "Teacher Monitoring + Awareness Materials"

        df['AI_RecommendedAction'] = df.apply(assign_action, axis=1)

        features = ['Age', 'GenderNum', 'AttendanceRate', 'AcademicPerformance',
                    'CounselingSessions', 'BehaviorIncidentsNum', 'HotlineCalls', 'RiskLevel']
        X = df[features]
        y_act = df['AI_RecommendedAction']

        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y_act, test_size=0.2, random_state=42, stratify=y_act
        )

        model_dt = DecisionTreeClassifier(random_state=42)
        model_dt.fit(X_train2, y_train2)

        y_pred2 = model_dt.predict(X_test2)
        print("DT Accuracy:", accuracy_score(y_test2, y_pred2))
        print("\nDT Classification Report:\n", classification_report(y_test2, y_pred2))

        df_test = X_test2.copy()
        df_test['AI_RecommendedAction'] = y_test2
        df_test['AI_PredictedAction'] = y_pred2
        df_test.to_csv(_ROOT / "students_test_predictions.csv", index=False)

        joblib.dump(model_dt, _ROOT / "decision_tree_model.pkl")
