import streamlit as st
from streamlit_utils import get_rag_components

st.title("Settings")

changed_settings = ""


#####################################################################
########################   SHOW SOURCES   ###########################
#####################################################################
v_show_sources = st.session_state.show_sources
show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
if show_sources != v_show_sources:
    changed_settings = "show_sources"
    st.session_state.show_sources = show_sources


#####################################################################
############################   LLM   ################################
#####################################################################
v_llm = st.session_state.llm
llm = st.selectbox("Which model to use?", ("ollama/llama3.1:8b", "ollama/qwen2.5:7b", "lm-studio", "gemini-flash-1.5"))
if llm != v_llm:
    changed_settings = "llm"
    st.session_state.llm = llm
    get_rag_components.clear()

if changed_settings != "":
    html_string = f"<p style='color: green;'>Settings changed: {changed_settings}</p>"
    st.markdown(html_string, unsafe_allow_html=True)