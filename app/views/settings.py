import streamlit as st

st.title(":rainbow[Settings]")
st.markdown("Welcome to the Settings page! Here you can change the settings for the app.")
st.markdown("---")

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
model_options = ("ollama/llama3.1:8b", "ollama/qwen2.5:7b", "lm-studio", "gemini-1.5-flash")
v_llm_index = model_options.index(v_llm)
llm = st.selectbox("Which model to use?", model_options, index=v_llm_index)
if llm != v_llm:
    changed_settings = "llm"
    st.session_state.llm = llm

if changed_settings != "":
    html_string = f"<p style='color: green;'>Settings changed: {changed_settings}</p>"
    st.markdown(html_string, unsafe_allow_html=True)
