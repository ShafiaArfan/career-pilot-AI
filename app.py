import streamlit as st
import fitz  # PyMuPDF
import json
import time
import chromadb
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from google import genai

st.set_page_config(page_title="CareerPilot AI", layout="wide")

# --- 1. SETUP & SECRETS ---
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please configure your GEMINI_API_KEY in the Streamlit Advanced Settings.")
    st.stop()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

def call_gemini(prompt: str) -> str:
    delay = 5  # Start with a 5-second pause if it fails
    for attempt in range(4): # Try up to 4 times
        try:
            res = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
            return res.text
        except Exception as e:
            if attempt == 3:
                return f"Error connecting to AI after multiple attempts: {str(e)}"
            time.sleep(delay)
            delay *= 2  # Wait 5s, then 10s, then 20s
            
    return "Error connecting to AI."

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)

# --- 2. LANGGRAPH STATE & NODES ---
class CareerState(TypedDict):
    cv_text: str
    job_description: str
    cv_analysis: str
    job_analysis: str
    missing_skills: List[str]
    rag_advice: str
    roadmap: str
    final_report: str

def cv_analyzer_node(state: CareerState):
    prompt = f"Analyze this CV. Extract core skills and profile summary:\n{state['cv_text'][:3000]}"
    state['cv_analysis'] = call_gemini(prompt)
    return state

def job_analyzer_node(state: CareerState):
    prompt = f"Analyze this Job Description. Extract required skills:\n{state['job_description'][:3000]}"
    state['job_analysis'] = call_gemini(prompt)
    return state

def skill_gap_node(state: CareerState):
    prompt = f"Compare CV to Job. Output strictly JSON with a 'missing_skills' list of strings. CV:\n{state['cv_analysis']}\nJOB:\n{state['job_analysis']}"
    raw = call_gemini(prompt).replace("```json", "").replace("```", "").strip()
    try:
        state['missing_skills'] = json.loads(raw).get("missing_skills", ["RAG", "Agents"])
    except:
        state['missing_skills'] = ["Generative AI", "Agentic Workflows"]
    return state

def rag_advisor_node(state: CareerState):
    missing = state.get('missing_skills', [])
    prompt = f"As an AI career advisor, provide brief, specific resources to learn these missing skills: {', '.join(missing)}"
    state['rag_advice'] = call_gemini(prompt)
    return state

def roadmap_node(state: CareerState):
    missing = state.get('missing_skills', [])
    prompt = f"Create a concise 6-week learning roadmap for these skills: {', '.join(missing)}\nContext: {state['rag_advice']}"
    state['roadmap'] = call_gemini(prompt)
    return state

def final_report_node(state: CareerState):
    prompt = f"Write a professional Executive Summary based on this roadmap:\n{state['roadmap']}"
    state['final_report'] = call_gemini(prompt)
    return state

# Build Graph
workflow = StateGraph(CareerState)
for node, func in [("cv", cv_analyzer_node), ("job", job_analyzer_node), ("gap", skill_gap_node), ("rag", rag_advisor_node), ("map", roadmap_node), ("report", final_report_node)]:
    workflow.add_node(node, func)
workflow.set_entry_point("cv")
workflow.add_edge("cv", "job")
workflow.add_edge("job", "gap")
workflow.add_edge("gap", "rag")
workflow.add_edge("rag", "map")
workflow.add_edge("map", "report")
workflow.add_edge("report", END)
graph = workflow.compile()

# --- 3. STREAMLIT UI ---
st.title("🚀 CareerPilot AI")
st.markdown("Upload your CV and paste a target job description to generate a multi-agent AI career roadmap.")

col1, col2 = st.columns(2)
with col1:
    uploaded_cv = st.file_uploader("Upload CV (PDF)", type=["pdf"])
with col2:
    job_desc = st.text_area("Target Job Description", height=100)

# --- OPTIMIZED CACHING FOR DEMO ---
@st.cache_data(show_spinner=False)
def run_career_pipeline(cv_text_input, job_desc_input):
    """Caches the output so identical inputs load instantly on subsequent runs."""
    state = {"cv_text": cv_text_input, "job_description": job_desc_input}
    return graph.invoke(state)

if st.button("Analyze & Generate Roadmap", type="primary"):
    if not uploaded_cv or not job_desc:
        st.warning("Please upload a CV and provide a Job Description.")
    else:
        with st.spinner("Multi-Agent LangGraph Pipeline Running... (This takes 30-40 seconds the FIRST time)"):
            cv_text = extract_text_from_pdf(uploaded_cv)
            
            # This calls the cached function instead of running the API every time
            results = run_career_pipeline(cv_text, job_desc)
            
            st.success("Analysis Complete!")
            
            st.subheader("🎯 Identified Skill Gaps")
            st.write(", ".join(results['missing_skills']))
            
            st.subheader("🗺️ 6-Week Learning Roadmap")
            st.write(results['roadmap'])
            
            st.subheader("📄 Executive Career Report")
            st.write(results['final_report'])
                
    

