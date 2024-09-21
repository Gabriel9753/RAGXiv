import streamlit as st

st.title("Settings")

changed_settings = ""


v_show_sources = True
if "show_sources" in st.session_state:
    v_show_sources = st.session_state.show_sources
else:
    st.session_state.show_sources = True
show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
if show_sources != v_show_sources:
    changed_settings = "show_sources"
st.session_state.show_sources = show_sources


if changed_settings != "":
    html_string = f"<p style='color: green;'>Settings changed: {changed_settings}</p>"
    st.markdown(html_string, unsafe_allow_html=True)