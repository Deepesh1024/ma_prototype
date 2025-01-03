import streamlit as st
from prediction import predictor
from langchain_groq import ChatGroq

# Import the *ChatPrompt* classes you actually need
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Create LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="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"
)

st.title("Blood pressure analyser")

# 1) Define your system message template
system_template = """
You are a professional medical assistant specializing in cardiovascular health. 
You analyze input data consisting of blood pressure readings (SYS/DIA), pulse rate, 
age, and cardiovascular probability (a percentage indicating the risk of cardiovascular disease). 
Based on this data, you provide a diagnosis-like explanation that highlights the user's health status, 
identifies potential concerns, and offers actionable recommendations.

Goals:
- Interpret the Data
- Generate Insights
- Recommend Next Steps

Rules:
- Always provide clear, concise, and medically accurate insights.
- Use a warm and non-alarming tone unless the data indicates an emergency.
- Avoid providing a definitive diagnosis or prescribing treatment; 
  instead, recommend consulting a healthcare professional for personalized care.
  
  
  keep it concise and to the point. Also return everything in modern hindi and new terms.
"""

system_prompt = SystemMessagePromptTemplate.from_template(system_template)

# 2) Define your human/user message template
# Notice we keep placeholders `{sys}`, `{dia}`, etc. to fill them in dynamically
human_template = (
    "My SYS is {sys}, my DIA is {dia}, my pulse is {pulse}, "
    "my cardiovascular score is {cardio} out of 1, and my age is 40."
)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 3) Combine the system and human messages into one chat prompt
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

# Create Streamlit UI
sys_val = st.number_input("Enter Systolic Blood Pressure (sys)", min_value=0, max_value=300, value=120)
dia_val = st.number_input("Enter Diastolic Blood Pressure (dia)", min_value=0, max_value=200, value=80)
pulse_val = st.number_input("Enter Pulse")

if st.button("Analyze"):
    # Suppose 'predictor(sys, dia)' returns a float that represents the 'cardio' score
    result = predictor(sys_val, dia_val)
    
    final_messages = chat_prompt.format_messages(
        sys=sys_val,
        dia=dia_val,
        pulse=pulse_val,
        cardio=result
    )
    
    response = llm.invoke(final_messages)
    st.write(response.content)