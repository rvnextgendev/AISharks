"""Streamlit interface for the AI Sharks panel."""
import asyncio
import json
import os
import time

import streamlit as st
from dotenv import load_dotenv
from groq import AuthenticationError, RateLimitError, APIConnectionError
from ai_sharks_structured import MODEL, SHARKS, pitch, run_panel

st.set_page_config(page_title="Pitch to the AI Sharks", page_icon="🦈", layout="wide")
load_dotenv()


with st.sidebar:
    st.title("🦈 AI Sharks")
    st.caption("Five perspectives. One stronger pitch.")
    key = st.text_input("Your Groq API key", type="password", help="Used for this session. Never included in reports.")
    configured_key = os.getenv("GROQ_API_KEY", "")
    try:
        configured_key = st.secrets.get("GROQ_API_KEY", configured_key)
    except st.errors.StreamlitSecretNotFoundError:
        pass
    key = key.strip() or configured_key.strip()
    st.caption("Key ready" if key else "Enter your key to evaluate a pitch.")
    st.caption(f"Model: {MODEL}")
    st.divider()
    st.markdown("**Meet the panel**")
    for shark in SHARKS:
        st.write(f"**{shark.name}** · {shark.title}")
    st.caption("Each evaluation makes 11 model calls before retries. Groq usage limits apply.")

st.title("Pitch to the AI Sharks")
st.write("Put your idea in front of five AI investors. Explore their first impressions, debate, and final offers.")
st.caption("AI simulation for pitch practice. Offers are fictional.")
mode = st.radio("Pitch source", ["Sample pitch", "Write a pitch", "Upload .txt"], horizontal=True)
text = ""
if mode == "Sample pitch":
    text = st.text_area("Edit the sample pitch", value=pitch, height=280)
elif mode == "Write a pitch":
    text = st.text_area("Your startup pitch", height=280, placeholder="Business name, problem, solution, customers, traction, revenue, costs, funding ask…")
else:
    uploaded = st.file_uploader("Upload a UTF-8 pitch file", type=["txt"])
    if uploaded is not None:
        if uploaded.size > 100_000:
            st.error("Choose a text file smaller than 100 KB.")
        else:
            try:
                text = st.text_area("Review and edit your pitch", value=uploaded.getvalue().decode("utf-8-sig"), height=280)
            except UnicodeDecodeError:
                st.error("Save the file using UTF-8 encoding, then upload it again.")

if st.button("Evaluate my pitch", type="primary"):
    if not key:
        st.error("Enter your Groq API key in the sidebar.")
    elif not text.strip():
        st.error("Enter or upload a pitch first.")
    elif len(text) > 20_000:
        st.error("Please shorten your pitch to 20,000 characters or fewer.")
    else:
        st.session_state.pop("panel_result", None)
        started = time.perf_counter()
        try:
            with st.spinner("The sharks are reviewing your pitch and discussing their decisions…"):
                result = asyncio.run(run_panel(text, api_key=key))
            st.session_state.panel_result = {
                "pitch": text,
                "round1": [v.model_dump() for v in result["round1"]],
                "final": [v.model_dump() for v in result["final"]],
                "summary": result["summary"],
                "seconds": round(time.perf_counter() - started, 1),
            }
        except AuthenticationError:
            st.error("Groq rejected the key. Check it and try again.")
        except RateLimitError:
            st.error("Groq's usage limit was reached. Wait before trying again.")
        except APIConnectionError:
            st.error("Could not connect to Groq. Check your connection and try again.")
        except Exception:
            st.error("Evaluation failed. Check your model access and package installation, then try again.")

result = st.session_state.get("panel_result")
if result:
    st.divider()
    if text != result["pitch"]:
        st.info("These results belong to the last evaluated pitch. Evaluate again to use your changes.")
    investing = sum(v["stance"] == "INVEST" for v in result["final"])
    a, b, c = st.columns(3)
    a.metric("Invest", investing)
    b.metric("Pass", len(SHARKS) - investing)
    c.metric("Elapsed", f"{result['seconds']} s")
    summary, final, first = st.tabs(["Host summary", "Final decisions", "First round"])
    with summary:
        st.subheader("The host's takeaway")
        st.write(result["summary"])
        with st.expander("Pitch evaluated"):
            st.text(result["pitch"])
    for tab, field in [(final, "final"), (first, "round1")]:
        with tab:
            for index, verdict in enumerate(result[field]):
                with st.container(border=True):
                    st.subheader(f"{verdict['shark']} · {SHARKS[index].title}")
                    if verdict["stance"] == "INVEST":
                        st.success("INVEST")
                    else:
                        st.info("PASS")
                    if verdict["offer"]:
                        st.write("**Offer:**", verdict["offer"])
                    st.write(verdict["reasoning"])
    st.download_button("Download evaluation (JSON)", json.dumps(result, indent=2, ensure_ascii=False), file_name="sharks_evaluation.json", mime="application/json")
