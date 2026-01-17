import streamlit as st
import requests

LANGSERVE_URL = "http://localhost:8000/llm/invoke"

# Page config
st.set_page_config(
    page_title="MultiMagix",
    page_icon="🎬",
    layout="centered"
)

# Header
#st.image("../images/logo.png", width=180)
st.title("MultiMagix")
st.caption("AI-powered Media Intelligence Platform by Maigha")

st.divider()

# Input
user_prompt = st.text_area(
    "Ask something about multimagix or media processing",
    height=120
)

if st.button("Run", type="primary"):
    if not user_prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    LANGSERVE_URL,
                    json={"input": user_prompt},
                    timeout=60
                )
                response.raise_for_status()

                result = response.json()["output"]

                st.success("Done")
                st.markdown("### Response")
                st.write(result)

            except Exception as e:
                st.error(f"Error: {e}")

st.divider()
st.caption("© Maigha • Powered with AI")