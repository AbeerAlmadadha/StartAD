import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import numpy as np
# -------------------------------
# 1- قراءة بيانات الملاحظات
# -------------------------------
# نقرأ من مجموعة بيانات الصحة النفسية التي تحتوي على الأعمدة: teacher_note, label
df_notes = pd.read_csv("data/teacher_notes_mental_health.csv", encoding="utf-8-sig")

# معالجة القيم الفارغة لتجنّب أخطاء في التوكنيزر؛ نضع نصاً افتراضياً للحالات الفارغة
df_notes['teacher_note'] = df_notes['teacher_note'].fillna('').astype(str).str.strip()
df_notes.loc[df_notes['teacher_note'] == '', 'teacher_note'] = 'عادي'

print(f"عدد العينات: {len(df_notes)}")
# -------------------------------
# 2- تحميل AraBERT
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
model.eval()  # وضع النموذج في وضع inference
# -------------------------------
# 3- تحويل النصوص إلى embeddings
# -------------------------------
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
    # نأخذ المتوسط كتمثيل للجملة
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.numpy()
X_embeddings = get_embeddings(df_notes['teacher_note'], tokenizer, model)
y = df_notes['label']
# -------------------------------
# 4- تقسيم البيانات
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_embeddings, y, test_size=0.2, random_state=42, stratify=y
)
# -------------------------------
# 5- تدريب Classifier على embeddings
# -------------------------------
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
# -------------------------------
# 6- تقييم النموذج
# -------------------------------
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# -------------------------------
# 7- التنبؤ لكل الطلاب وحفظ النتائج
# -------------------------------
df_notes['RiskLevel'] = clf.predict(X_embeddings)
df_notes[['RiskLevel']].to_csv("teacher_risk_levels_arabert.csv", index=True)
print("تم حفظ التقييم في teacher_risk_levels_arabert.csv")
