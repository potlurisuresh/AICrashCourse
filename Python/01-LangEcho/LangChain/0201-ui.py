import streamlit as st
import importlib

mmapp = importlib.import_module("0201-mmapp")

mmapp.init_llm()

# Page config
st.set_page_config(
    page_title="Maigha",
    layout="centered"
)

# Header
#st.image("../images/logo.png", width=180)
st.title("Maigha")
st.caption("AI-powered Intelligence Platform by Maigha")

st.divider()

# Input
user_prompt = st.text_area(
    "Ask something",
    height=120
)

# Action button
if st.button("Run", type="primary"):
    with st.spinner("Thinking..."):
        try:
            response = mmapp.generate_response(user_prompt)
            st.success("Done")
            st.markdown("### Response")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

# Footer
st.caption("© Maigha • Powered with AI")
