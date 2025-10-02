# combined.py

import streamlit as st
import re
import torch
import joblib
from transformers import AutoTokenizer, AutoModel

# ---------------------------
# 1️⃣ Streamlit Page Config
# ---------------------------
st.set_page_config(page_title="أداة تقييم المخاطر النفسية للطلاب", layout="wide")

# ---------------------------
# 2️⃣ Load Models
# ---------------------------
@st.cache_resource
def load_models():
    # Load AraBERT
    tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
    bert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2")
    bert_model.eval()

    # Load Logistic Regression for RiskLevel
    lr_model = joblib.load("saved_model/logistic_regression_risk.pkl")
    
    # Load Decision Tree for Action Recommendation
    dt_model = joblib.load("saved_model/decision_tree_action.pkl")
    
    return tokenizer, bert_model, lr_model, dt_model

tokenizer, bert_model, lr_model, dt_model = load_models()

# ---------------------------
# 3️⃣ Preprocess Arabic Text
# ---------------------------
def preprocess_arabic_text(text):
    if not text or text.strip() == "":
        return ""
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)
    return text

# ---------------------------
# 4️⃣ Predict Risk Level
# ---------------------------
def predict_risk(note):
    clean_note = preprocess_arabic_text(note)
    if not clean_note:
        return "none", 1.0, {"none": 1.0}
    
    # Get AraBERT embeddings
    inputs = tokenizer([clean_note], return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).numpy()
    
    # Logistic Regression prediction
    pred = lr_model.predict(embedding)[0]
    prob = lr_model.predict_proba(embedding)[0]
    class_probs = dict(zip(lr_model.classes_, prob))
    confidence = max(prob)
    return pred, confidence, class_probs

# ---------------------------
# 5️⃣ Predict AI Recommended Action
# ---------------------------
def predict_action(features):
    return dt_model.predict([features])[0]

# ---------------------------
# 6️⃣ Streamlit UI
# ---------------------------
st.markdown(
    """
    <div style="text-align:center; padding:20px; background-color:#d73527; border-radius:12px;">
        <h1 style="color:white; margin-bottom:5px;">🧠 أداة تقييم المخاطر النفسية للطلاب</h1>
        <p style="color:white; font-size:16px;">تحديد الطلاب المعرضين لمخاطر الصحة النفسية وتقديم التدخل المناسب</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Input section
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🆔 معلومات الطالب")
    student_id = st.text_input("رقم الطالب:")
    age = st.number_input("العمر:", min_value=5, max_value=25, value=15)
    gender = st.selectbox("الجنس:", ["M", "F"])
    gender_num = 0 if gender == "M" else 1

with col2:
    st.markdown("### ⚠️ ملاحظات مثيرة للقلق فقط")
    prompt = st.text_area(
        "أدخل أي سلوك أو موقف مثير للقلق (اتركه فارغاً إذا لم تكن هناك مخاوف):",
        height=150,
        placeholder="مثال: الطالب يبدو حزيناً باستمرار، أو يتجنب التفاعل مع الآخرين، أو يظهر تغيرات سلوكية مفاجئة..."
    )
    attendance_rate = st.number_input("نسبة الحضور:", min_value=0, max_value=100, value=80)
    academic_perf = st.number_input("الأداء الأكاديمي:", min_value=0, max_value=100, value=70)
    counseling_sessions = st.number_input("عدد جلسات الإرشاد:", min_value=0, value=0)
    hotline_calls = st.number_input("عدد المكالمات لخط المساعدة:", min_value=0, value=0)
    behavior_incident = st.selectbox("الحوادث السلوكية:", ["سلوك طبيعي", "معتدل", "عنيف"])
    behavior_num = {"سلوك طبيعي":0, "معتدل":1, "عنيف":2}[behavior_incident]

st.write("")
submit = st.button("🔍 تقييم المخاطر النفسية", key="submit")

# ---------------------------
# 7️⃣ Handle Prediction
# ---------------------------
if submit:
    if not student_id:
        st.warning("⚠️ يرجى إدخال رقم الطالب.")
    else:
        # 1- Predict RiskLevel
        risk, confidence, probs = predict_risk(prompt)
        
        # Map RiskLevel to numeric for decision tree
        risk_map = {"low":0, "moderate":1, "high":2, "none":-1}
        risk_num = risk_map.get(risk, -1)
        
        # 2- Build features for Decision Tree
        features = [age, gender_num, attendance_rate, academic_perf, counseling_sessions, behavior_num, hotline_calls, risk_num]
        ai_action = predict_action(features)
        
        # ---------------------------
        # Display RiskLevel
        # ---------------------------
        risk_colors = {"none":"#28a745", "low":"#ffc107", "moderate":"#fd7e14", "high":"#dc3545"}
        risk_labels = {"none":"لا توجد مخاطر", "low":"مخاطر منخفضة", "moderate":"مخاطر متوسطة", "high":"مخاطر عالية"}
        selected_color = risk_colors.get(risk, "#6c757d")
        selected_label = risk_labels.get(risk, "غير معروف")
        
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#f8f9fa; padding:20px; border-radius:12px; border-left:5px solid {selected_color};">
                    <h3 style="color:{selected_color};">� تقييم المخاطر النفسية للطالب رقم: {student_id}</h3>
                    <p><strong>مستوى المخاطر:</strong> {selected_label}</p>
                    <p><strong>الثقة في التنبؤ:</strong> {confidence:.1%}</p>
                    <p><strong>توصية AI:</strong> {ai_action}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Add emergency resources if high risk
        if risk == "high":
            st.markdown(
                """
                <div style="background-color:#f8d7da; padding:20px; border-radius:8px; border:1px solid #f5c6cb; margin-top:10px;">
                    <h4 style="color:#721c24;">🚨 موارد الطوارئ</h4>
                    <p style="color:#721c24;"><strong>خط المساعدة النفسية:</strong> 920033360</p>
                    <p style="color:#721c24;"><strong>الطوارئ:</strong> 997</p>
                    <p style="color:#721c24;"><strong>وزارة الصحة:</strong> 937</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:10px; color:#6c757d; font-size:12px;">
        <p><strong>تنبيه:</strong> هذه الأداة للمساعدة فقط ولا تغني عن الاستشارة النفسية المتخصصة</p>
        <p>في حالات الطوارئ، اتصل فوراً بالأخصائي النفسي أو الطوارئ</p>
    </div>
    """,
    unsafe_allow_html=True
)
