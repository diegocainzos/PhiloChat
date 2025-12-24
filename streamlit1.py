import streamlit as st

st.info('This is a purely informational message', icon="ℹ️")
text_area = st.text_area('enter text', '')
empty = st.empty()
with st.form("my_form")  :
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    but = st.form_submit_button() 
    if but:
        
        empty.text('pulsaste el boton')
        