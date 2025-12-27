import uuid
import streamlit as st
if "my_session_id" not in st.session_state:
    # Generamos un ID único la primera vez que carga la página
    st.session_state.my_session_id = str(uuid.uuid4())

# Al llamar a LangChain, le pasamos este ID
respuesta = chain.invoke(
    {"input": pregunta},
    config={"configurable": {"session_id": st.session_state.my_session_id}}
)