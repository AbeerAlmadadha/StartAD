# source .venv/bin/activate
# streamlit run app.py


import streamlit as st
from final import generate_ai_suggestions

# Page config
st.set_page_config(page_title="أداة دعم اتخاذ القرار للمعلمين", layout="wide")

# Header
st.markdown(
    """
    <div style="text-align:center; padding:20px; background-color:#2c5282; border-radius:12px;">
        <h1 style="color:white; margin-bottom:5px;">🏫 أداة دعم اتخاذ القرار للمعلمين</h1>
        <p style="color:white; font-size:16px;">مساعدة المعلمين في التعرف على مشاكل الطلاب وتقديم الدعم في الوقت المناسب</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Input section in two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🆔 معلومات الطالب")
    student_id = st.text_input("رقم الطالب:")

with col2:
    st.markdown("### ✏️ ملاحظات المعلم")
    prompt = st.text_area("أدخل ملاحظاتك هنا:", height=180)

st.write("")

# Submit button
submit = st.button("إنشاء اقتراحات الذكاء الاصطناعي", key="submit")

# AI Response area
if submit:
    if not student_id or not prompt:
        st.warning("⚠️ يرجى ملء رقم الطالب والملاحظات قبل الإرسال.")
    else:
        with st.spinner("جارٍ تحليل الملاحظة وتوليد التوصيات..."):
            try:
                result = generate_ai_suggestions(student_id, prompt)
            except Exception as e:
                st.error("حدث خطأ أثناء توليد الاقتراحات. يرجى المحاولة لاحقًا.")
                st.exception(e)
                result = None

        if result:
            st.balloons()
            # AI response card
            with st.container():
                st.markdown(
                    f"""
                    <div style="
                        background-color:#f0f4f8;
                        padding:25px;
                        border-radius:12px;
                        border-left:5px solid #2c5282;
                        box-shadow: 2px 2px 12px rgba(0,0,0,0.1);
                        color: #000000;
                    ">
                        <h3 style="color:#2c5282;">🤖 اقتراحات الذكاء الاصطناعي للطالب رقم: {result.get('student_id','')}</h3>
                        <p><strong>مستوى الخطورة:</strong> {result.get('risk_level','-')}</p>
                        <p><strong>الإجراء الموصى به:</strong> {result.get('recommended_action','-')}</p>
                        <details>
                            <summary>عرض التفاصيل</summary>
                            <div style="margin-top:10px;">
                                <p><strong>الأسباب/المؤشرات:</strong></p>
                                <ul>
                                    {''.join(f'<li>{item}</li>' for item in result.get('rationale', []))}
                                </ul>
                                <p><strong>إجراءات فورية:</strong></p>
                                <ul>
                                    {''.join(f'<li>{item}</li>' for item in result.get('immediate_actions', []))}
                                </ul>
                                <p><strong>متابعة لاحقة:</strong></p>
                                <ul>
                                    {''.join(f'<li>{item}</li>' for item in result.get('follow_up', []))}
                                </ul>
                            </div>
                        </details>
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
