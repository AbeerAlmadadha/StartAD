# source .venv/bin/activate

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


import streamlit as st

# Page config
st.set_page_config(page_title="Staff Decision-Support Tool", layout="wide")

# Header
st.markdown(
    """
    <div style="text-align:center; padding:20px; background-color:#2c5282; border-radius:12px;">
        <h1 style="color:white; margin-bottom:5px;">🏫 Staff Decision-Support Tool</h1>
        <p style="color:white; font-size:16px;">Helping teachers identify student wellness issues and provide timely support</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")

# Input section in two columns
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 🆔 Student Info")
    student_id = st.text_input("Student ID:")
    st.markdown("### ✏️ Observation Type")
    observation_type = st.selectbox("Select type of concern:", ["Mild", "Moderate", "Severe"])

with col2:
    st.markdown("### ✏️ Teacher Observation")
    prompt = st.text_area("Enter your observations here:", height=180)

st.write("")

# Submit button
submit = st.button("Generate AI Suggestions", key="submit")

# AI Response area
if submit:
    if not student_id or not prompt:
        st.warning("⚠️ Please fill in Student ID and observations before submitting.")
    else:
        # Balloon effect
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
                ">
                    <h3 style="color:#2c5282;">🤖 AI Suggestions for Student ID: {student_id}</h3>
                    <p><b>Observation Type:</b> {observation_type}</p>
                    <details>
                        <summary>View AI Recommendations</summary>
                        <p style="margin-top:10px;">(This is where AI output will appear. You can integrate LangChain/OpenAI here.)</p>
                    </details>
                </div>
                """,
                unsafe_allow_html=True
            )
