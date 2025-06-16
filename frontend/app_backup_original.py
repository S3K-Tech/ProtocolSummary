import streamlit as st
import requests
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import logging
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from backend.utils.token_utils import count_tokens, calculate_cost
from backend.utils.export_utils import create_word_doc, create_pdf
from backend.core.config import OPENAI_API_KEY, GROQ_API_KEY
from backend.core.drug_info_extractor import DrugInfoExtractor

# Inject custom CSS for UI enhancements
css_path = Path(__file__).parent / "static" / "css" / "custom_ui.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Enhanced Modern Tab Styling with Optimized Sidebar
st.markdown("""
<style>
/* Sidebar Optimization */
.block-container {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

.element-container {
    margin-bottom: 0.5rem;
}

/* Compact section headers */
.section-heading {
    font-size: 1rem !important;
    font-weight: 600;
    color: #1e293b;
    margin: 0.5rem 0 0.3rem 0 !important;
    padding: 0.3rem 0 !important;
    border-bottom: 1px solid #e2e8f0;
}

/* Modern Results Section Styling */
.results-header {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    border: 2px solid #cbd5e1;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
}
.results-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e293b;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}
.results-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Enhanced Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    padding: 0.5rem;
    border-radius: 12px;
    border: 2px solid #cbd5e1;
    margin: 1.5rem 0 1rem 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background: transparent;
    border-radius: 8px;
    padding: 0.8rem 1.5rem;
    font-weight: 600;
    font-size: 1rem;
    color: #64748b;
    transition: all 0.3s ease;
    border: none;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}

.stTabs [data-baseweb="tab"]:hover {
    background: rgba(59, 130, 246, 0.1);
    color: #3b82f6;
    transform: translateY(-1px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"]::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 50%;
    transform: translateX(-50%);
    width: 60%;
    height: 3px;
    background: #fbbf24;
    border-radius: 2px;
}

/* Enhanced Tab Content Area */
.stTabs [data-baseweb="tab-panel"] {
    border: 2px solid #e2e8f0;
    border-top: none;
    padding: 2rem;
    border-radius: 0 0 12px 12px;
    background: #ffffff;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    min-height: 400px;
    position: relative;
}

.stTabs [data-baseweb="tab-panel"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #06b6d4 100%);
    border-radius: 0 0 8px 8px;
}

/* Generate Button Enhancement */
.generate-section {
    margin: 2rem 0;
    text-align: center;
}

/* Upload Section Styling */
.upload-section {
    border: 2px dashed #cbd5e1;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8fafc;
    transition: all 0.3s ease;
}

.upload-section:hover {
    border-color: #3b82f6;
    background: #eff6ff;
}

.upload-file-info {
    background: #e0f2fe;
    border-radius: 6px;
    padding: 0.5rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
    color: #0f172a;
}

/* Document Type Headers */
.doc-type-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 600;
    color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

# Ensure .env variables are loaded for API keys
load_dotenv()

# Enhanced Chatbot Functions
def render_document_chatbot():
    """Enhanced chatbot with web search capability"""
    
    # Add border around entire chatbot section
    st.sidebar.markdown("""
    <div style='border: 2px solid #ddd; border-radius: 8px; padding: 10px; margin: 5px 0; background-color: #f8f9fa;'>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 💬 Medical Writer Assistant")
    
    # Search type selection
    search_type = st.sidebar.radio(
        "Search in:",
        ["📄 Documents", "🌐 Web"],
        horizontal=True,
        key="chatbot_search_type"
    )
    
    search_mapping = {
        "📄 Documents": "documents",
        "🌐 Web": "web"
    }
    search_mode = search_mapping[search_type]
    
    # Document selection
    selected_collections = []
    if search_mode == "documents":
        st.sidebar.markdown("**Select Documents:**")
        
        doc_options = {
            "PT": "pt-index", "RP": "rp-index", 
            "PS": "ps-index", "IB": "ib-index"
        }
        
        selected_docs = st.sidebar.multiselect(
            "",
            options=list(doc_options.keys()),
            default=list(doc_options.keys()),  # Select all documents by default
            key="chatbot_docs"
        )
        selected_collections = [doc_options[doc] for doc in selected_docs]
    
    # Web search filter
    web_search_type = "medical"
    if search_mode == "web":
        web_search_type = st.sidebar.selectbox(
            "Web search focus:",
            ["medical", "pubmed", "general"],
            key="chatbot_web_type"
        )
    

    
    # Model selection
    model_options = {
        "GPT-3.5": {"provider": "openai", "model": "gpt-3.5-turbo"},
        "GPT-4": {"provider": "openai", "model": "gpt-4-turbo"},
        "Llama 3 🆓": {"provider": "groq", "model": "llama3-8b-8192"}
    }
    
    selected_model = st.sidebar.selectbox(
        "AI Model:",
        options=list(model_options.keys()),
        index=0,
        key="chatbot_model"
    )
    model_info = model_options[selected_model]
    
    # Chat history with limit
    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = []
    
    # Limit chat history to last 50 messages (25 exchanges) to prevent memory issues
    if len(st.session_state.chatbot_history) > 50:
        st.session_state.chatbot_history = st.session_state.chatbot_history[-50:]
    
    # Recent chat display
    if st.session_state.chatbot_history:
        st.sidebar.markdown("**Recent Chat:**")
        # Show only last exchange (question + answer)
        recent_messages = st.session_state.chatbot_history[-2:] if len(st.session_state.chatbot_history) >= 2 else st.session_state.chatbot_history
        
        for msg in recent_messages:
            role_emoji = "👤" if msg["role"] == "user" else "🤖"
            role_name = "You" if msg["role"] == "user" else "Assistant"
            
            st.sidebar.markdown(f"**{role_emoji} {role_name}:**")
            
            # Display content in scrollable container
            if msg["role"] == "assistant":
                st.sidebar.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 8px; border-radius: 5px; margin: 5px 0; max-height: 100px; overflow-y: auto; font-size: 0.85em;'>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.sidebar.markdown(f"""
                <div style='background-color: #e8f4f8; padding: 8px; border-radius: 5px; margin: 5px 0; max-height: 80px; overflow-y: auto; font-size: 0.85em;'>
                    {msg['content']}
                </div>
                """, unsafe_allow_html=True)
            
            # Sources display
            if msg["role"] == "assistant":
                sources_count = len(msg.get("sources", [])) + len(msg.get("web_sources", []))
                if sources_count > 0:
                    with st.sidebar.expander(f"📚 {sources_count} sources"):
                        # Document sources
                        if msg.get("sources"):
                            st.markdown("**📄 Documents:**")
                            for i, source in enumerate(msg["sources"]):
                                st.markdown(f"• {source}")
                        
                        # Web sources
                        if msg.get("web_sources"):
                            st.markdown("**🌐 Web:**")
                            for source in msg["web_sources"]:
                                st.markdown(f"• [{source['title']}]({source['link']})")
    
    # Chat input
    user_query = st.sidebar.text_area(
        "Ask your question:",
        value=st.session_state.get("chatbot_query_input", ""),
        height=68,
        placeholder="Type your question about the protocol documents...",
        key="chatbot_input"
    )
    
    if user_query != st.session_state.get("chatbot_query_input", ""):
        st.session_state.chatbot_query_input = user_query
    
    # Action buttons
    col1, col2 = st.sidebar.columns([2, 1])
    
    with col1:
        send_clicked = st.button("Send", use_container_width=True, key="chatbot_send", type="primary")
    with col2:
        clear_clicked = st.button("Clear", use_container_width=True, key="chatbot_clear")
    
    # Handle send button
    if send_clicked and user_query and (selected_collections or search_mode == "web"):
        temp_query = user_query
        st.session_state.chatbot_query_input = ""
        
        # Add user message
        st.session_state.chatbot_history.append({"role": "user", "content": temp_query})
        
        with st.sidebar.status("Searching..."):
            try:
                # Call backend API
                response = requests.post(
                    "http://localhost:8000/chatbot/chat",
                    json={
                        "query": temp_query,
                        "collections": selected_collections,
                        "search_type": search_mode,
                        "provider": model_info["provider"],
                        "model": model_info["model"],
                        "web_search_type": web_search_type
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.chatbot_history.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": data.get("sources", []),
                        "web_sources": data.get("web_sources", [])
                    })
                else:
                    st.session_state.chatbot_history.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error. Please try again.",
                        "sources": [],
                        "web_sources": []
                    })
            except Exception as e:
                st.session_state.chatbot_history.append({
                    "role": "assistant", 
                    "content": f"Error: {str(e)}",
                    "sources": [],
                    "web_sources": []
                })
        
        st.rerun()
    
    # Handle clear button
    if clear_clicked:
        st.session_state.chatbot_history = []
        st.session_state.chatbot_query_input = ""
        st.rerun()
    
    # Full conversation view (collapsible)
    if st.session_state.chatbot_history:
        exchanges_count = len(st.session_state.chatbot_history)//2
        with st.sidebar.expander(f"Full Chat ({exchanges_count} exchanges)"):
            for msg in st.session_state.chatbot_history:
                role_emoji = "👤" if msg["role"] == "user" else "🤖"
                role_name = "You" if msg["role"] == "user" else "Assistant"
                
                st.markdown(f"**{role_emoji} {role_name}:**")
                
                # Display content in scrollable container
                if msg["role"] == "assistant":
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 8px; border-radius: 5px; margin: 5px 0; max-height: 200px; overflow-y: auto; font-size: 0.9em;'>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background-color: #e8f4f8; padding: 8px; border-radius: 5px; margin: 5px 0; max-height: 150px; overflow-y: auto; font-size: 0.9em;'>
                        {msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                
                if msg["role"] == "assistant":
                    # Document sources
                    if "sources" in msg and msg["sources"]:
                        st.markdown(f"**📄 Document Sources ({len(msg['sources'])})**")
                        for i, source in enumerate(msg["sources"]):
                            st.markdown(f"  {i+1}. {source}")
                    
                    # Web sources
                    if "web_sources" in msg and msg["web_sources"]:
                        st.markdown(f"**🌐 Web Sources ({len(msg['web_sources'])})**")
                        for source in msg["web_sources"]:
                            st.markdown(f"  • [{source['title']}]({source['link']})")
                    
                    st.markdown("---")
    
    # Close chatbot border
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

# Custom loader function
def show_custom_loader():
    st.markdown('''<div class="custom-loader"><div class="dna-loader"><span></span><span></span><span></span><span></span></div></div>''', unsafe_allow_html=True)

# --- Enhanced Main Header Section ---
st.markdown("""
    <style>
    .main-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
        padding: 1.5rem 1rem;
        margin: 1rem auto 0.5rem auto;
        max-width: 1400px;
        border: none;
        position: relative;
        overflow: hidden;
    }
    .main-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="medical" patternUnits="userSpaceOnUse" width="40" height="40"><circle cx="20" cy="20" r="1.5" fill="rgba(255,255,255,0.1)"/><path d="M10,20 L30,20 M20,10 L20,30" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100" height="100" fill="url(%23medical)"/></svg>');
        opacity: 0.4;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: 0.5px;
        text-align: center;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        margin: 0;
        position: relative;
        z-index: 2;
        line-height: 1.1;
    }
    .main-subtitle {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-top: 0.3rem;
        font-weight: 300;
        position: relative;
        z-index: 2;
        letter-spacing: 0.3px;
    }
    .header-stats {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        margin-top: 0.8rem;
        position: relative;
        z-index: 2;
    }
    .stat-item {
        text-align: center;
        color: rgba(255, 255, 255, 0.95);
    }
    .stat-number {
        font-size: 1.4rem;
        font-weight: 700;
        display: block;
        color: #fbbf24;
    }
    .stat-label {
        font-size: 0.9rem;
        font-weight: 400;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-card">
    <div class="main-title">🏥 AI Powered Clinical Trial Protocol Generator</div>
    <div class="main-subtitle">Advanced AI-driven protocol section generation with semantic search and medical expertise</div>
    <div class="header-stats">
        <div class="stat-item">
            <span class="stat-number">20+</span>
            <span class="stat-label">Section Templates</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">ICH</span>
            <span class="stat-label">M11 Compliant</span>
        </div>
        <div class="stat-item">
            <span class="stat-number">AI</span>
            <span class="stat-label">Powered Analysis</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# Section heading
st.markdown('<div class="section-heading"><span class="icon">📄</span>Protocol Section</div>', unsafe_allow_html=True)

# Fetch available section templates from backend (with caching)
if 'section_templates' not in st.session_state:
try:
        # Add timeout and better error handling
        templates_resp = requests.get(
            "http://localhost:8000/available_section_templates",
            timeout=10  # 10 second timeout
        )
    templates_resp.raise_for_status()
    section_templates = templates_resp.json()
        
        # Cache the templates in session state
        st.session_state['section_templates'] = section_templates
        
    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please check if the backend server is running and try refreshing the page.")
        st.session_state['section_templates'] = []
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend server. Please ensure the backend is running on http://localhost:8000")
        st.session_state['section_templates'] = []
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e.response.status_code} - {e.response.text}")
        st.session_state['section_templates'] = []
    except Exception as e:
        st.error(f"❌ Failed to fetch section templates: {e}")
        st.session_state['section_templates'] = []
else:
    section_templates = st.session_state['section_templates']

# Process templates
if section_templates and len(section_templates) > 0:
    section_options = {item["title"]: item["key"] for item in section_templates}
    section_title = st.selectbox("Select Protocol Section", list(section_options.keys()))
    section_key = section_options[section_title]
    
    # Get template details for the selected section
    template_details = next((t for t in section_templates if t["key"] == section_key), None)
    default_prompt = f"Generate a {section_title} section for a clinical trial protocol."
else:
    st.warning("No section templates available. Using default section.")
    st.info("💡 If you're seeing this, try refreshing the page or check the backend connection.")
    section_key = "default"
    section_title = "Default Section"
    default_prompt = "Generate a clinical trial protocol section."

# Section divider
st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# Section heading
st.markdown('<div class="section-heading"><span class="icon">📝</span>Additional Instructions</div>', unsafe_allow_html=True)

user_prompt = st.text_area(
    "Additional Instructions (optional)",
    value="",
    height=100,
    help="Add any specific instructions or requirements for this section.",
    placeholder="Enter any specific instructions or requirements for this section..."
)

# Section divider
st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)

# Enhanced Results Section Header
st.markdown("""
<div class="results-header">
    <div class="results-title">
        <span>📊</span> Protocol Generation Results
    </div>
    <div class="results-subtitle">
        Generate, review, and analyze your clinical trial protocol sections with AI assistance
    </div>
</div>
""", unsafe_allow_html=True)

# Custom loader markup (function defined above)


with st.sidebar:
    # --- Document Upload & Indexing ---
    st.markdown('<div class="section-heading"><span class="icon">📄</span>Document Management</div>', unsafe_allow_html=True)
    
    # Document type configuration
    doc_types = {
        'PS': {'name': 'Protocol Summaries', 'icon': '📋', 'color': '#3b82f6'},
        'PT': {'name': 'Protocol Templates', 'icon': '📄', 'color': '#10b981'},
        'RP': {'name': 'Reference Protocols', 'icon': '📚', 'color': '#f59e0b'},
        'IB': {'name': 'Investigator\'s Brochure', 'icon': '🔬', 'color': '#ef4444'}
    }
    
    for doc_type, config in doc_types.items():
        with st.expander(f"{config['icon']} {config['name']} ({doc_type})", expanded=False):
            # File uploader
            uploaded_files = st.file_uploader(
                f"Upload {doc_type} documents",
                type=['pdf', 'docx'],
                accept_multiple_files=True,
                key=f"{doc_type}_upload",
                help=f"Select one or more PDF or DOCX files for {config['name']}"
            )
            
            # Show file info
            if uploaded_files:
                st.write(f"**Selected files ({len(uploaded_files)}):**")
                for file in uploaded_files:
                    file_size = f"{file.size:,} bytes" if file.size else "Unknown size"
                    st.write(f"• {file.name} ({file_size})")
            
            # Upload and index button
            if uploaded_files and st.button(f"📤 Upload & Index {doc_type}", key=f"{doc_type}_index", type="primary"):
                with st.spinner(f"Processing {doc_type} documents..."):
            try:
                        # Prepare files for upload
                        files = []
                        for file in uploaded_files:
                            files.append(("files", (file.name, file.getvalue(), file.type)))
                        
                        # Send to backend
                        response = requests.post(
                            "http://localhost:8000/upload_and_index",
                            files=files,
                            data={"collection_type": doc_type}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.success(f"✅ {data['message']}")
                            st.info(f"Files processed: {', '.join(data['files'])}")
                else:
                            st.error(f"❌ Upload failed: {response.text}")
                            
            except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            # Show empty state
            if not uploaded_files:
                st.info(f"No {doc_type} files selected. Choose files above to upload.")

    # --- Step 2: Model Settings (Compact) ---
    st.markdown('<div class="section-heading"><span class="icon">🤖</span>Model Settings</div>', unsafe_allow_html=True)
    # Simple model lists (updated to match token_utils.py)
    openai_models = {
        "GPT-4": {"provider": "openai", "id": "gpt-4"},
        "GPT-4 Turbo": {"provider": "openai", "id": "gpt-4-turbo"},
        "GPT-4o": {"provider": "openai", "id": "gpt-4o"},
        "GPT-4o Mini": {"provider": "openai", "id": "gpt-4o-mini"}
    }
    groq_models = {
        "🆓 Llama 3.3 70B": {"provider": "groq", "id": "llama-3.3-70b-versatile"},
        "🆓 Llama 3.1 8B": {"provider": "groq", "id": "llama-3.1-8b-instant"},
        "🆓 Llama 3 70B": {"provider": "groq", "id": "llama3-70b-8192"},
        "🆓 Llama 3 8B": {"provider": "groq", "id": "llama3-8b-8192"},
        "🆓 Gemma 2 9B": {"provider": "groq", "id": "gemma2-9b-it"}
    }
    provider = st.radio("Provider", ["OpenAI", "Groq"], key="provider_radio", horizontal=True)
    if provider == "OpenAI":
        openai_model_name = st.selectbox("Model", list(openai_models.keys()), key="openai_model_select")
        st.session_state["selected_model"] = openai_models[openai_model_name]["id"]
        st.session_state["provider"] = openai_models[openai_model_name]["provider"]
    elif provider == "Groq":
        groq_model_name = st.selectbox("Model", list(groq_models.keys()), key="groq_model_select")
        st.session_state["selected_model"] = groq_models[groq_model_name]["id"]
        st.session_state["provider"] = groq_models[groq_model_name]["provider"]

    # --- Step 3: Admin Controls (Compact) ---
    with st.expander("🛠️ Model Controls"):
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=1, step=1)
        st.session_state["temperature"] = temperature
        st.session_state["top_k"] = top_k

    # --- Medical Writer Assistant Chatbot ---
    render_document_chatbot()

# Compact Footer
st.markdown(
    """
    <div style='margin-top:1em;text-align:center;color:#b0b8c1;font-size:0.8rem;'>
        <span>v1.0 | AI Powered</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Initialize session state variables if they don't exist
if 'generated_section' not in st.session_state:
    st.session_state['generated_section'] = None
if 'section_title' not in st.session_state:
    st.session_state['section_title'] = None
if 'section_key' not in st.session_state:
    st.session_state['section_key'] = None
if 'show_review' not in st.session_state:
    st.session_state['show_review'] = False
if 'section_approved' not in st.session_state:
    st.session_state['section_approved'] = False
if 'chunks_info' not in st.session_state:
    st.session_state['chunks_info'] = []
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = ''

# 2. Reference Chunk Selection (auto-retrieval only)
selected_chunks = {}  # Always empty; backend will select relevant chunks

# --- Main Content Area with Tabs ---
# Create tabs for different sections of the output/workflow
main_tab, reference_tab = st.tabs(["Section Content", "Reference Information"])

with main_tab:
    # Enhanced Generate Section button
    st.markdown('<div class="generate-section">', unsafe_allow_html=True)
    generate_clicked = st.button("🚀 Generate Protocol Section", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if generate_clicked:
        if not section_key:
            st.error("No section template selected. Cannot generate section.")
        else:
            with st.spinner("Generating section..."):
                try:
                    # Use default prompt if no additional instructions provided
                    final_prompt = user_prompt.strip() if user_prompt.strip() else default_prompt
                    
                    req = {
                        "user_prompt": final_prompt,
                        "selected_chunks": selected_chunks,
                        "section_key": section_key,
                        "selected_model": st.session_state.get('selected_model', 'gpt-4-turbo'),
                        "provider": st.session_state.get('provider', 'openai'),
                        "temperature": st.session_state.get('temperature', 0.2),
                        "top_k": st.session_state.get('top_k', 2)
                    }
                    resp = requests.post("http://localhost:8000/generate_section_with_chunks", json=req)
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if "error" in data:
                        st.error(f"Error from backend: {data['error']}")
                    else:
                        section = data.get("output", "")
                        prompt = data.get("prompt", "")
                        chunks_info = data.get("chunks_info", [])
                        
                        if not section or section.startswith("[ERROR]"):
                            st.error(section if section else "No content was generated. Please try again with different parameters.")
                        else:
                            # Store the generated section and related info in session state
                            st.session_state['generated_section'] = section
                            st.session_state['section_title'] = section_title
                            st.session_state['section_key'] = section_key
                            st.session_state['show_review'] = True # Show review interface
                            st.session_state['chunks_info'] = chunks_info # Store chunks_info for reference tab
                            st.session_state['prompt'] = prompt # Store prompt for reference tab
                            st.session_state['section_approved'] = False # Reset approval status on new generation
                            st.session_state['edited_section_content'] = section # Initialize edited content
                            
                            # Clear any previous analysis cache when new section is generated
                            analysis_keys_to_clear = [
                                'generated_drugs', 'reference_drugs', 'drug_comparison', 'extraction_method',
                                'grammar_analysis', 'consistency_analysis'
                            ]
                            for key in analysis_keys_to_clear:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            st.rerun()

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to backend: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during generation: {str(e)}")
                    logging.error(f"Error in frontend generation: {e}", exc_info=True)

    # --- Section Review Area (Conditional Display) ---
    # Show review interface, approval, and history if a section is generated
    if st.session_state.get('show_review', False):
        st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
        st.markdown('<div class="section-heading"><span class="icon">✍️</span>Section Review</div>', unsafe_allow_html=True)
        
        # Conditional display: editable text area if not approved, static text if approved
        current_section_content = st.session_state.get('edited_section_content', '') # Content to use for display/analysis

        if not st.session_state.get('section_approved', False):
            edited_section_content = st.text_area(
                "Review and Edit Section",
                value=current_section_content, # Use content from state
                height=400,
                help="Review the generated section. Make any necessary edits before approval.",
                key='editable_review_area' # Add a unique key
            )
            # Update the edited_section_content in session state whenever text area changes
            st.session_state['edited_section_content'] = edited_section_content
            current_section_content = edited_section_content # Update for immediate display/analysis
            
            reviewer_comments = st.text_area(
                "Review Comments (Optional)",
                help="Add any comments or notes about the section",
                key='review_comments_area' # Add a unique key
            )

            # Approval button
            if st.button("✅ Approve Section", type="primary"):
                # Prepare review data using content from session state
                review_data = {
                    "section_key": st.session_state.get('section_key'),
                    "original_content": st.session_state.get('generated_section'),
                    "edited_content": st.session_state.get('edited_section_content'), # Use content from state
                    "status": "approved",
                    "reviewer_comments": reviewer_comments,
                    "version": 1, # Assuming this is the first approval version
                    "timestamp": datetime.now().isoformat()
                }
                
                # Submit for approval
                try:
                    response = requests.post(
                        "http://localhost:8000/section/approve",
                        json=review_data
                    )
                    
                    if response.status_code == 200:
                        st.session_state['approved_section'] = st.session_state.get('edited_section_content') # Store approved content
                        st.session_state['section_approved'] = True
                        st.success("Section approved! You can now proceed with analysis and export.")
                        st.rerun()
                    else:
                        logging.error(f"Backend approval failed: {response.status_code} - {response.text}")
                        st.error("Failed to approve section. Please try again.")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to backend for approval: {str(e)}")
                except Exception as e:
                    st.error(f"An unexpected error occurred during approval: {str(e)}")
                    logging.error(f"Error in frontend approval: {e}", exc_info=True)

        else: # Section is approved, show static content
            st.markdown("### Approved Section Content")
            st.markdown(st.session_state.get('approved_section', ''))
            current_section_content = st.session_state.get('approved_section', '') # Use approved content for display/analysis
            # Optionally display review comments for the approved version here
            # st.markdown(f"**Review Comments:** {reviewer_comments}") # Need to store/retrieve comments with approved version

        # --- Token/Cost Info --- # Display token/cost information after review/approval
        if current_section_content and st.session_state.get('prompt'):
            model_used = st.session_state.get('selected_model', 'gpt-4')
            provider_used = st.session_state.get('provider', 'openai')
            
            input_tokens = count_tokens(st.session_state.get('prompt', ''), model_used)
            output_tokens = count_tokens(current_section_content, model_used)
            
            # Use new cost calculation system
            total_cost, cost_breakdown = calculate_cost(
                input_tokens,
                output_tokens,
                model_used,
                provider_used
            )
            
            provider_label = "OpenAI" if provider_used.lower().startswith("openai") else ("Groq" if provider_used.lower().startswith("groq") else provider_used)
            
            # Create cost display with breakdown
            if cost_breakdown.get('is_free_tier', False):
                cost_display = f"<span style='color:#28a745;font-weight:bold;'>FREE</span> <small style='color:#666;'>({cost_breakdown.get('note', 'Free tier')})</small>"
            else:
                cost_display = f"<span style='color:#dc3545;'>${total_cost:.6f}</span>"
                if cost_breakdown.get('pricing_source') == 'manual':
                    cost_display += " <small style='color:#ffc107;'>⚠️ Legacy pricing</small>"
            
            # Check for deprecated models
            from backend.utils.token_utils import is_model_deprecated, get_model_replacement
            deprecated_warning = ""
            if is_model_deprecated(model_used):
                replacement = get_model_replacement(model_used)
                deprecated_warning = f"<br/><small style='color:#dc3545;'>⚠️ Model deprecated. Consider switching to: {replacement}</small>"
            
            st.markdown(f"""
                <div style='background:#f7fafd;border-radius:8px;padding:0.7em 1em 0.7em 1em;margin-bottom:0.7em;border:1px solid #e0e3e8;'>
                <b>Model:</b> {model_used} ({provider_label}) &nbsp;|&nbsp; 
                <b>Tokens:</b> {input_tokens + output_tokens:,} (In: {input_tokens:,}, Out: {output_tokens:,}) &nbsp;|&nbsp; 
                <b>Cost:</b> {cost_display}
                {deprecated_warning}
                </div>
            """, unsafe_allow_html=True)

        # --- Analysis Section (Conditional Display after Approval) ---
        if st.session_state.get('section_approved', False):
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            st.markdown('<div class="section-heading"><span class="icon">🔍</span>Section Analysis</div>', unsafe_allow_html=True)
            
            # Create tabs for different analyses
            analysis_tab1, analysis_tab2 = st.tabs(["Key Drug Information", "Grammar & Consistency"])

            with analysis_tab1:
                if st.button("Analyze Drug Information", key='analyze_drug_btn'):
                    with st.spinner("Analyzing drug information using AI-free extraction..."):
                         try:
                            analysis_req = {
                                "text": current_section_content,
                                "reference_text": st.session_state.get('prompt', ''),
                                "selected_model": st.session_state.get('selected_model', ''),
                                "provider": st.session_state.get('provider', ''),
                                "temperature": st.session_state.get('temperature', 0.2)
                            }
                            analysis_resp = requests.post("http://localhost:8000/analyze/drugs", json=analysis_req)
                            analysis_resp.raise_for_status()
                            analysis_data = analysis_resp.json()
                            
                            # Store results in session state
                            st.session_state['generated_drugs'] = analysis_data.get('generated_drugs', [])
                            st.session_state['reference_drugs'] = analysis_data.get('reference_drugs', [])
                            st.session_state['drug_comparison'] = analysis_data.get('comparison', None)
                            st.session_state['extraction_method'] = analysis_data.get('extraction_method', 'regex')
                            
                            st.success("Drug information analysis complete!")
                            st.rerun()
                         except requests.exceptions.RequestException as e:
                              st.error(f"Failed to connect to backend for drug analysis: {str(e)}")
                         except Exception as e:
                             st.error(f"An unexpected error occurred during drug analysis: {str(e)}")
                             logging.error(f"Error in frontend drug analysis: {e}", exc_info=True)

                # Display extracted drug information in a simple, clean way
                if st.session_state.get('generated_drugs'):
                    generated_drugs = st.session_state['generated_drugs']
                    
                    # Drug Comparison Analytics
                    reference_drugs = st.session_state.get('reference_drugs', [])
                    comparison = st.session_state.get('drug_comparison', None)
                    
                    st.markdown("### 💊 Drug Information Comparison")
                    
                    if comparison and reference_drugs:
                        # Comparison metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #0066cc;'>
                                    <div style='font-size: 2rem; color: #0066cc; font-weight: bold;'>{len(reference_drugs)}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>In Reference</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #28a745;'>
                                    <div style='font-size: 2rem; color: #28a745; font-weight: bold;'>{len(comparison.get('matches', []))}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>Matches</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #dc3545;'>
                                    <div style='font-size: 2rem; color: #dc3545; font-weight: bold;'>{len(comparison.get('missing', []))}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>Missing</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #17a2b8;'>
                                    <div style='font-size: 2rem; color: #17a2b8; font-weight: bold;'>{len(comparison.get('additional', []))}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>Additional</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        # Simple Analytics (when no reference comparison is available)
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #0066cc;'>
                                    <div style='font-size: 2rem; color: #0066cc; font-weight: bold;'>{len(generated_drugs)}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>Drugs Found</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            drugs_with_dosage = len([d for d in generated_drugs if d.get('dosage')])
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #28a745;'>
                                    <div style='font-size: 2rem; color: #28a745; font-weight: bold;'>{drugs_with_dosage}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>With Dosage</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            drugs_with_frequency = len([d for d in generated_drugs if d.get('frequency')])
                            st.markdown(f"""
                                <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid #17a2b8;'>
                                    <div style='font-size: 2rem; color: #17a2b8; font-weight: bold;'>{drugs_with_frequency}</div>
                                    <div style='font-size: 0.9rem; color: #666;'>With Frequency</div>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Extraction Method Info
                    st.markdown("### 🔧 Extraction Details")
                    method = st.session_state.get('extraction_method', 'regex')
                    st.markdown(f"""
                        <div style='background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #28a745; margin-bottom: 1rem;'>
                            <strong>Method:</strong> {method.title()} Pattern Recognition | 
                            <strong>Source:</strong> Generated Protocol Section
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Comparison Results Display
                    if comparison and reference_drugs:
                        # Show comprehensive comparison results
                        st.markdown("### 📋 Drug Information Comparison Results")
                        
                        # Summary
                        total_reference = len(reference_drugs)
                        matches = len(comparison.get('matches', []))
                        missing = len(comparison.get('missing', []))
                        additional = len(comparison.get('additional', []))
                        
                        if total_reference > 0:
                            coverage = (matches / total_reference) * 100
                            if coverage >= 80:
                                st.success(f"✅ Excellent coverage! Generated content includes {matches}/{total_reference} drugs from reference ({coverage:.0f}%).")
                            elif coverage >= 60:
                                st.warning(f"⚠️ Good coverage. Generated content includes {matches}/{total_reference} drugs from reference ({coverage:.0f}%).")
                            else:
                                st.error(f"❌ Low coverage. Generated content includes only {matches}/{total_reference} drugs from reference ({coverage:.0f}%).")
                        
                        # Missing Drugs (Priority Alert for Medical Writer)
                        if missing > 0:
                            st.markdown("### 🔴 Missing from Generated Content (Action Required)")
                            st.error(f"The following {missing} drugs from the reference documents were NOT found in the generated content:")
                            
                            missing_data = []
                            for drug in comparison['missing']:
                                missing_data.append({
                                    "Drug Name": drug.get('name', ''),
                                    "Dosage": drug.get('dosage', '—'),
                                    "Frequency": drug.get('frequency', '—'),
                                    "Route": drug.get('route', '—'),
                                    "Form": drug.get('form', '—'),
                                    "Reference Context": drug.get('source_text', '')
                                })
                            
                            df_missing = pd.DataFrame(missing_data)
                            st.dataframe(df_missing, use_container_width=True, hide_index=True)
                        
                        # Matched Drugs
                        if matches > 0:
                            with st.expander(f"✅ Matched Drugs ({matches} drugs)", expanded=False):
                                match_data = []
                                for match in comparison['matches']:
                                    ref_drug = match['reference']
                                    gen_drug = match['generated']
                                    match_data.append({
                                        "Drug Name": match['name'],
                                        "Reference Dosage": ref_drug.get('dosage', '—'),
                                        "Generated Dosage": gen_drug.get('dosage', '—'),
                                        "Reference Frequency": ref_drug.get('frequency', '—'), 
                                        "Generated Frequency": gen_drug.get('frequency', '—'),
                                        "Reference Context": ref_drug.get('source_text', ''),
                                        "Generated Context": gen_drug.get('source_text', ''),
                                        "Status": "✅ Perfect Match" if match.get('details_match', False) else "⚠️ Partial Match"
                                    })
                                
                                df_matches = pd.DataFrame(match_data)
                                st.dataframe(df_matches, use_container_width=True, hide_index=True)
                        
                        # Additional Drugs
                        if additional > 0:
                            with st.expander(f"➕ Additional Drugs Not in Reference ({additional} drugs)", expanded=False):
                                additional_data = []
                                for drug in comparison['additional']:
                                    additional_data.append({
                                        "Drug Name": drug.get('name', ''),
                                        "Dosage": drug.get('dosage', '—'),
                                        "Frequency": drug.get('frequency', '—'),
                                        "Route": drug.get('route', '—'),
                                        "Form": drug.get('form', '—'),
                                        "Generated Context": drug.get('source_text', '')
                                    })
                                
                                df_additional = pd.DataFrame(additional_data)
                                st.dataframe(df_additional, use_container_width=True, hide_index=True)
                    
                    else:
                        # Fallback: Simple drug information display when no reference comparison
                        st.markdown("### 📋 Extracted Drug Information")
                        
                        if generated_drugs:
                            # Create a clean table like the original output
                            drug_table_data = []
                            for drug in generated_drugs:
                                drug_table_data.append({
                                    "Drug Name": drug.get('name', ''),
                                    "Dosage": drug.get('dosage', '—'),
                                    "Frequency": drug.get('frequency', '—'),
                                    "Route": drug.get('route', '—'),
                                    "Form": drug.get('form', '—'),
                                    "Generated Context": drug.get('source_text', '')
                                })
                            
                            df = pd.DataFrame(drug_table_data)
                            st.dataframe(df, use_container_width=True, hide_index=True)
                        else:
                            st.info("No drugs were found in the generated protocol section.")

                else:
                    # Welcome message
                    st.markdown("""
                    ### 💊 Drug Information Analysis
                    
                    Compare drug information between your generated protocol section and reference documents:
                    
                    - **🔍 Smart Pattern Recognition:** Advanced regex patterns for medical terminology
                    - **⚡ Fast Processing:** Instant results with no API calls
                    - **📊 Reference Comparison:** Check if generated content includes drugs from reference documents
                    - **⚠️ Missing Drug Alerts:** Identify what the LLM missed from reference materials
                    
                    Click the button below to analyze and compare drug information.
                    """)
                    st.info("👆 Click 'Analyze Drug Information' to compare drug information between your generated content and reference documents.")

            with analysis_tab2:
                # Grammar & Consistency Analysis
                if st.button("Analyze Grammar & Consistency", key='analyze_grammar_btn'):
                    with st.spinner("Analyzing grammar, consistency, and ICH M11 compliance..."):
                        try:
                            # Prepare analysis request
                            analysis_req = {
                                "text": current_section_content,
                                "section_name": st.session_state.get('section_title', ''),
                                "reference_text": st.session_state.get('prompt', '')
                            }
                            
                            # Call backend API
                            analysis_resp = requests.post("http://localhost:8000/analyze/grammar", json=analysis_req)
                            analysis_resp.raise_for_status()
                            analysis_data = analysis_resp.json()
                            
                            # Store results in session state
                            st.session_state['grammar_analysis_results'] = analysis_data.get('analysis', {})
                            
                            st.success("Grammar & Consistency analysis complete!")
                            st.rerun()
                            
                        except requests.exceptions.RequestException as e:
                            st.error(f"Failed to connect to backend: {str(e)}")
                        except Exception as e:
                            st.error(f"Analysis failed: {str(e)}")
                            logging.error(f"Error in grammar analysis: {e}", exc_info=True)
                
                # Display analysis results if available
                if st.session_state.get('grammar_analysis_results'):
                    results = st.session_state['grammar_analysis_results']
                    
                    # Display scores with visual indicators
                    st.markdown("### 📊 Quality Scores")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        score = results.get('scores', {}).get('grammar', 0)
                        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid {color};'>
                                <div style='font-size: 2rem; color: {color}; font-weight: bold;'>{score:.0f}</div>
                                <div style='font-size: 0.9rem; color: #666;'>Grammar</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        score = results.get('scores', {}).get('consistency', 0)
                        color = "green" if score >= 80 else "orange" if score >= 60 else "red" 
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid {color};'>
                                <div style='font-size: 2rem; color: {color}; font-weight: bold;'>{score:.0f}</div>
                                <div style='font-size: 0.9rem; color: #666;'>Consistency</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        score = results.get('scores', {}).get('compliance', 0)
                        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid {color};'>
                                <div style='font-size: 2rem; color: {color}; font-weight: bold;'>{score:.0f}</div>
                                <div style='font-size: 0.9rem; color: #666;'>ICH Compliance</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        score = results.get('scores', {}).get('overall', 0)
                        color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                        st.markdown(f"""
                            <div style='text-align: center; padding: 1rem; border-radius: 8px; background: #f8f9fa; border: 2px solid {color};'>
                                <div style='font-size: 2rem; color: {color}; font-weight: bold;'>{score:.0f}</div>
                                <div style='font-size: 0.9rem; color: #666;'>Overall</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Summary
                    st.markdown("### 📋 Analysis Summary")
                    st.info(results.get('summary', 'No summary available'))
                    
                    # Issues found
                    issues = results.get('issues', [])
                    if issues:
                        st.markdown(f"### 🔍 Issues Found ({len(issues)})")
                        
                        # Group issues by severity
                        high_issues = [i for i in issues if i.get('severity') == 'high']
                        medium_issues = [i for i in issues if i.get('severity') == 'medium']
                        low_issues = [i for i in issues if i.get('severity') == 'low']
                        
                        # Display high priority issues first
                        if high_issues:
                            st.markdown("#### 🔴 High Priority")
                            for issue in high_issues:
                                with st.expander(f"[{issue.get('type', '').upper()}] {issue.get('message', '')}", expanded=True):
                                    st.write(f"**💡 Suggestion:** {issue.get('suggestion', '')}")
                                    if issue.get('location'):
                                        st.markdown(issue.get('location', ''))
                                    if issue.get('category'):
                                        st.write(f"**🏷️ Category:** {issue.get('category', '')}")
                        
                        # Medium priority issues
                        if medium_issues:
                            st.markdown("#### 🟡 Medium Priority")
                            for issue in medium_issues:
                                with st.expander(f"[{issue.get('type', '').upper()}] {issue.get('message', '')}"):
                                    st.write(f"**💡 Suggestion:** {issue.get('suggestion', '')}")
                                    if issue.get('location'):
                                        st.markdown(issue.get('location', ''))
                                    if issue.get('category'):
                                        st.write(f"**🏷️ Category:** {issue.get('category', '')}")
                        
                        # Low priority issues
                        if low_issues:
                            st.markdown("#### 🟢 Low Priority")
                            for issue in low_issues:
                                with st.expander(f"[{issue.get('type', '').upper()}] {issue.get('message', '')}"):
                                    st.write(f"**💡 Suggestion:** {issue.get('suggestion', '')}")
                                    if issue.get('location'):
                                        st.markdown(issue.get('location', ''))
                                    if issue.get('category'):
                                        st.write(f"**🏷️ Category:** {issue.get('category', '')}")
                    else:
                        st.success("✅ No issues found! The section meets high standards for medical writing.")
                    
                    # Recommendations
                    recommendations = results.get('recommendations', [])
                    if recommendations:
                        st.markdown("### 💡 Recommendations")
                        for rec in recommendations:
                            st.write(f"• {rec}")
                
                else:
                    st.info("Click 'Analyze Grammar & Consistency' to check grammar, terminology consistency, and ICH M11 compliance.")

        # Show export options if section is approved
        if st.session_state.get('section_approved', False):
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:0.7em;'><b>Export:</b></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                word_doc = create_word_doc(st.session_state.get('approved_section', ''), st.session_state.get('section_title', ''))
                st.download_button(
                    label="Download as Word",
                    data=word_doc,
                    file_name=f"{st.session_state.get('section_title', '').replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            with col2:
                pdf = create_pdf(st.session_state.get('approved_section', ''), st.session_state.get('section_title', ''))
                st.download_button(
                    label="Download as PDF",
                    data=pdf,
                    file_name=f"{st.session_state.get('section_title', '').replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )

with reference_tab:
    # --- Reference Information Tab (Conditional Display) ---
    # Display chunks and context only if a section is generated
    if st.session_state.get('show_review', False):
        # Display Retrieved Chunks table
        if st.session_state.get('chunks_info'):
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            st.markdown("### Retrieved Chunks")
            chunks_data = []
            for chunk in st.session_state.get('chunks_info', []):
                source = chunk.get("source", "Unknown")
                file_name = source.replace("-index", "").upper() if source != "Unknown" and "-index" in source else "Unknown"
                chunks_data.append({
                    "File Name": file_name,
                    "Score": f"{chunk.get('score', 0):.4f}",
                    "Source": source,
                    "Referred Data": chunk.get("content", "")
                })
            
            if chunks_data:
                st.dataframe(
                    pd.DataFrame(chunks_data),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No chunks were retrieved for this generation.")

        # Display LLM Context (Reference Content Sent to LLM)
        if st.session_state.get('prompt'):
            st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
            st.markdown("### LLM Context (Reference Content Sent to LLM)") # Changed heading
            st.expander("Show Context").write(st.session_state.get('prompt', '') if st.session_state.get('prompt', '') else "No reference content was used.")


