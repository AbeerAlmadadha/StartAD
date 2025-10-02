# source .venv/bin/activate
# streamlit run app.py

import streamlit as st
import pickle
import re
import sys
import os

# Add parent directory to path to import model files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(page_title="أداة تقييم المخاطر النفسية للطلاب", layout="wide")

# Load model function
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'saved_model')
        with open(os.path.join(model_path, 'classifier.pkl'), 'rb') as f:
            classifier = pickle.load(f)
        with open(os.path.join(model_path, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        return classifier, vectorizer
    except:
        return None, None

# Preprocess Arabic text
def preprocess_arabic_text(text):
    if not text or text.strip() == "":
        return ""
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    text = arabic_diacritics.sub('', text)
    return text

# Predict mental health risk
def predict_mental_health_risk(note, classifier, vectorizer):
    if not note or note.strip() == "":
        return "none", 1.0, {"none": 1.0, "low": 0.0, "moderate": 0.0, "high": 0.0}

    clean_note = preprocess_arabic_text(note)
    if not clean_note:
        return "none", 1.0, {"none": 1.0, "low": 0.0, "moderate": 0.0, "high": 0.0}

    note_tfidf = vectorizer.transform([clean_note])
    prediction = classifier.predict(note_tfidf)[0]
    probabilities = classifier.predict_proba(note_tfidf)[0]
    confidence = max(probabilities)

    class_probs = dict(zip(classifier.classes_, probabilities))
    return prediction, confidence, class_probs

# Header
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

# Important notice
# st.markdown(
#     """
#     <div style="background-color:#fff3cd; padding:15px; border-radius:8px; border-left:5px solid #ffc107;">
#         <h4 style="color:#856404; margin-bottom:10px;">📋 تعليمات الاستخدام</h4>
#         <p style="color:#856404; margin-bottom:5px;">• أدخل <strong>فقط المواقف المثيرة للقلق</strong> أو السلوكيات المشكوك فيها</p>
#         <p style="color:#856404; margin-bottom:5px;">• اترك الحقل فارغاً إذا لم تكن هناك مخاوف</p>
#         <p style="color:#856404; margin-bottom:0px;">• لا تدخل تعليقات إيجابية أو عادية عن الطالب</p>
#     </div>
#     """,
#     unsafe_allow_html=True
# )

st.write("")

# Input section
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🆔 معلومات الطالب")
    student_id = st.text_input("رقم الطالب:")

with col2:
    st.markdown("### ⚠️ ملاحظات مثيرة للقلق فقط")
    prompt = st.text_area(
        "أدخل أي سلوك أو موقف مثير للقلق (اتركه فارغاً إذا لم تكن هناك مخاوف):",
        height=150,
        placeholder="مثال: الطالب يبدو حزيناً باستمرار، أو يتجنب التفاعل مع الآخرين، أو يظهر تغيرات سلوكية مفاجئة..."
    )

st.write("")

# Submit button
submit = st.button("🔍 تقييم المخاطر النفسية", key="submit")

# Load model
classifier, vectorizer = load_model()

# AI Response area
if submit:
    if not student_id:
        st.warning("⚠️ يرجى إدخال رقم الطالب.")
    else:
        if classifier is None:
            st.error("❌ خطأ في تحميل النموذج. يرجى التحقق من ملفات النموذج.")
        else:
            # Get prediction
            prediction, confidence, probabilities = predict_mental_health_risk(prompt, classifier, vectorizer)

            # Color coding for risk levels
            risk_colors = {
                "none": "#28a745",     # Green
                "low": "#ffc107",      # Yellow
                "moderate": "#fd7e14", # Orange
                "high": "#dc3545"      # Red
            }

            risk_labels = {
                "none": "لا توجد مخاطر",
                "low": "مخاطر منخفضة",
                "moderate": "مخاطر متوسطة",
                "high": "مخاطر عالية"
            }

            risk_recommendations = {
                "none": "✅ لا يحتاج الطالب لتدخل خاص حالياً. استمر في المتابعة العادية.",
                "low": "🟡 راقب الطالب واحرص على تقديم الدعم الإضافي والتشجيع.",
                "moderate": "🟠 يُنصح بالتواصل مع أولياء الأمور والأخصائي النفسي المدرسي.",
                "high": "🔴 تدخل فوري مطلوب! اتصل بالأخصائي النفسي وأولياء الأمور على الفور."
            }

            selected_color = risk_colors[prediction]
            selected_label = risk_labels[prediction]
            selected_recommendation = risk_recommendations[prediction]

            # Display results
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f8f9fa;
                        padding:25px;
                        border-radius:12px;
                        border-left:5px solid {selected_color};
                        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                    ">
                        <h3 style="color:{selected_color};">� تقييم المخاطر النفسية للطالب رقم: {student_id}</h3>

                        <div style="background-color:white; padding:15px; border-radius:8px; margin:15px 0;">
                            <h4 style="color:{selected_color}; margin-bottom:10px;">مستوى المخاطر: {selected_label}</h4>
                            <p style="font-size:16px; margin-bottom:10px;"><strong>الثقة في التنبؤ:</strong> {confidence:.1%}</p>

                            <div style="margin:15px 0;">
                                <h5>توزيع احتمالات المخاطر:</h5>
                                <ul style="list-style-type:none; padding-left:0;">
                                    <li>🟢 لا توجد مخاطر: {probabilities.get('none', 0):.1%}</li>
                                    <li>🟡 مخاطر منخفضة: {probabilities.get('low', 0):.1%}</li>
                                    <li>🟠 مخاطر متوسطة: {probabilities.get('moderate', 0):.1%}</li>
                                    <li>🔴 مخاطر عالية: {probabilities.get('high', 0):.1%}</li>
                                </ul>
                            </div>
                        </div>

                        <div style="background-color:#e9ecef; padding:15px; border-radius:8px;">
                            <h5 style="color:#495057; margin-bottom:10px;">💡 التوصيات:</h5>
                            <p style="color:#495057; margin-bottom:0px;">{selected_recommendation}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Add additional resources for high risk cases
            if prediction == "high":
                st.markdown(
                    """
                    <div style="background-color:#f8d7da; padding:20px; border-radius:8px; border:1px solid #f5c6cb; margin-top:20px;">
                        <h4 style="color:#721c24;">🚨 موارد الطوارئ</h4>
                        <p style="color:#721c24;"><strong>خط المساعدة النفسية:</strong> 920033360</p>
                        <p style="color:#721c24;"><strong>الطوارئ:</strong> 997</p>
                        <p style="color:#721c24;"><strong>وزارة الصحة:</strong> 937</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Footer with disclaimer
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








# import streamlit as st
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain_openai import ChatOpenAI  # ✅ New import

# st.title("Staff Decision-Support Tool")
# st.divider()
# st.write("""Enter your observations or concerns about a student.
#          The AI will suggest possible referral pathways to NCMH and ways to support the student.""")

# prompt = st.text_input("Your prompt:")
# st.divider()

# if prompt:
#     st.balloons()

    # # ✅ Use ChatOpenAI instead of old OpenAI
    # llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

    # # Prompt template
    # template = """You are an assistant helping teachers identify wellness issues.
    # Given a teacher's observation about a student, suggest:
    # 1. Possible referral pathways to the National Center for Mental Health (NCMH).
    # 2. Supportive actions the teacher/school can take immediately.
    # Keep responses clear, empathetic, and professional.

    # Teacher's note: {user_prompt}"""

    # prompt_template = PromptTemplate(input_variables=["user_prompt"], template=template)

    # # Build chain
    # llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    # # Run chain with teacher’s prompt
    # response = llm_chain.run(prompt)

    # st.subheader("Assistant Response:")
    # st.write(response)
