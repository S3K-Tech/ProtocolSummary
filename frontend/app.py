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
from backend.utils.export_utils import create_word_doc, create_pdf, sanitize_filename
from backend.core.config import OPENAI_API_KEY, GROQ_API_KEY
from backend.core.drug_info_extractor import DrugInfoExtractor

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Clinical Protocol Generator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for UI enhancements
css_path = Path(__file__).parent / "static" / "css" / "custom_ui.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Enhanced Modern Styling
st.markdown("""
<style>
/* Main Layout */
.block-container {
    padding-top: 1rem;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

.element-container {
    margin-bottom: 0.5rem;
}

/* Page Headers */
.page-header {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #1e40af 100%);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
}

.page-title {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.page-subtitle {
    font-size: 1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
}

/* Card Styling */
.info-card {
    background: #f8fafc;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}

/* Section Headers */
.section-heading {
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1rem 0 0.5rem 0;
    padding: 0.5rem 0;
    border-bottom: 2px solid #e2e8f0;
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

/* Cost Tracking */
.cost-info {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border: 2px solid #bbf7d0;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    text-align: center;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    padding: 0.5rem;
    border-radius: 12px;
    border: 2px solid #cbd5e1;
    margin: 1rem 0;
}

.stTabs [data-baseweb="tab"] {
    height: 48px;
    background: transparent;
    border-radius: 8px;
    padding: 0.8rem 1.5rem;
    font-weight: 600;
    color: #64748b;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
}

.stTabs [data-baseweb="tab-panel"] {
    border: 2px solid #e2e8f0;
    border-top: none;
    padding: 2rem;
    border-radius: 0 0 12px 12px;
    background: #ffffff;
    min-height: 400px;
}
</style>
""", unsafe_allow_html=True)

# ========================================
# UTILITY FUNCTIONS
# ========================================

def get_session_state_defaults():
    """Initialize session state with default values"""
    defaults = {
        'current_page': 'Document Management',
        'section_templates': [],
        'generated_section': None,
        'section_title': None,
        'section_key': None,
        'show_review': False,
        'section_approved': False,
        'chunks_info': [],
        'prompt': '',
        'edited_section_content': '',
        'chatbot_history': [],
        'chatbot_query_input': '',
        'generated_drugs': [],
        'reference_drugs': [],
        'drug_comparison': None,
        'grammar_analysis': None,
        'selected_model': 'llama-3.3-70b-versatile',
        'provider': 'groq',
        'temperature': 0.2,
        'top_k': 1,
        'total_tokens_used': 0,
        'total_cost': 0.0,
        'show_export_modal': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def show_cost_info(model, tokens_in, tokens_out, cost):
    """Display cost information in a standardized format"""
    st.markdown(f"""
    <div class="cost-info">
        💰 <strong>{model}</strong> | 
        Tokens: {tokens_in + tokens_out:,} (In: {tokens_in:,}, Out: {tokens_out:,}) | 
        Cost: {'FREE' if cost == 0 else f'${cost:.4f}'}
    </div>
    """, unsafe_allow_html=True)


# ========================================
# PAGE FUNCTIONS
# ========================================

def render_document_management():
    """Page 1: Document Management"""
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📄 Document Management</div>
        <div class="page-subtitle">Upload and manage your clinical trial documents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Document type configuration
    doc_types = {
        'PS': {'name': 'Protocol Summaries', 'icon': '📋', 'color': '#3b82f6'},
        'PT': {'name': 'Protocol Templates', 'icon': '📄', 'color': '#10b981'},
        'RP': {'name': 'Reference Protocols', 'icon': '📚', 'color': '#f59e0b'},
        'IB': {'name': 'Investigator\'s Brochure', 'icon': '🔬', 'color': '#ef4444'}
    }
    
    # Document Collections Grid
    st.markdown('<div class="section-heading">📚 Document Collections</div>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    for i, (doc_type, config) in enumerate(doc_types.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="info-card" style="text-align: center; border-color: {config['color']};">
                <h3>{config['icon']} {doc_type}</h3>
                <p><strong>{config['name']}</strong></p>
                <p>Documents: <strong>0</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
    # Upload Interface
    st.markdown('<div class="section-heading">📤 Upload Documents</div>', unsafe_allow_html=True)
    
    for doc_type, config in doc_types.items():
        with st.expander(f"{config['icon']} Upload {config['name']} ({doc_type})", expanded=False):
            uploaded_files = st.file_uploader(
                f"Upload {doc_type} documents",
                type=['pdf', 'docx'],
                accept_multiple_files=True,
                key=f"{doc_type}_upload",
                help=f"Select one or more PDF or DOCX files for {config['name']}"
            )
            
            if uploaded_files:
                st.write(f"**Selected files ({len(uploaded_files)}):**")
                for file in uploaded_files:
                    file_size = f"{file.size:,} bytes" if file.size else "Unknown size"
                    st.write(f"• {file.name} ({file_size})")
                
                if st.button(f"📤 Upload & Index {doc_type}", key=f"{doc_type}_index", type="primary"):
                    with st.spinner(f"Processing {doc_type} documents..."):
                        try:
                            files = []
                            for file in uploaded_files:
                                files.append(("files", (file.name, file.getvalue(), file.type)))
                            
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
            else:
                st.info(f"No {doc_type} files selected. Choose files above to upload.")
    
    # Quick Actions
    st.markdown('<div class="section-heading">⚡ Quick Actions</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start Generation", use_container_width=True, type="primary"):
            st.session_state.current_page = "Protocol Generator"
            st.rerun()
    with col2:
        if st.button("💬 Ask AI Assistant", use_container_width=True):
            st.session_state.current_page = "AI Assistant"
            st.rerun()
    with col3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.current_page = "Settings"
            st.rerun()
    
def render_protocol_generator():
    """Page 2: Protocol Generator"""
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🚀 Protocol Section Generator</div>
        <div class="page-subtitle">Generate AI-powered clinical trial protocol sections</div>
    </div>
    """, unsafe_allow_html=True)

    # Fetch section templates (with caching)
    if not st.session_state.section_templates:
        try:
            templates_resp = requests.get("http://localhost:8000/available_section_templates", timeout=10)
            templates_resp.raise_for_status()
            st.session_state.section_templates = templates_resp.json()
        except Exception as e:
            st.error(f"❌ Failed to fetch section templates: {e}")
            st.session_state.section_templates = []
    
    # Step 1: Section Selection
    st.markdown('<div class="section-heading">📄 Step 1: Select Section Type</div>', unsafe_allow_html=True)
    
    if st.session_state.section_templates:
        section_options = {item["title"]: item["key"] for item in st.session_state.section_templates}
        section_title = st.selectbox("Protocol Section", list(section_options.keys()))
        section_key = section_options[section_title]
        
        st.info(f"✓ Documents available for context retrieval")
    else:
        st.warning("No section templates available. Using default section.")
        section_key = "default"
        section_title = "Default Section"
    
    # Step 2: Additional Instructions
    st.markdown('<div class="section-heading">📝 Step 2: Additional Instructions (Optional)</div>', unsafe_allow_html=True)
    user_prompt = st.text_area(
        "Additional Instructions",
        value="",
        height=100,
        placeholder="Enter any specific instructions or requirements for this section..."
    )

    # Step 3: Generation Settings
    st.markdown('<div class="section-heading">⚙️ Step 3: Generation Settings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Model:** {st.session_state.selected_model}")
    with col2:
        st.write(f"**Temperature:** {st.session_state.temperature}")
    with col3:
        st.write(f"**Top-K:** {st.session_state.top_k}")
    
    # Generate Button
    st.markdown('<div class="section-heading">🎯 Generate Section</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Generate Section", type="primary", use_container_width=True):
        with st.spinner("Generating protocol section..."):
            try:
                generation_req = {
                    "section_key": section_key,
                    "user_prompt": user_prompt,
                    "selected_chunks": {},  # Empty dict for auto-retrieval
                    "selected_model": st.session_state.selected_model,
                    "provider": st.session_state.provider,
                    "temperature": st.session_state.temperature,
                    "top_k": st.session_state.top_k
                }
                
                response = requests.post("http://localhost:8000/generate_section_with_chunks", json=generation_req)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    st.error(f"Error: {data['error']}")
                else:
                    section = data.get("output", "")
                    if section and not section.startswith("[ERROR]"):
                        # Store generation results
                        st.session_state.generated_section = section
                        st.session_state.section_title = section_title
                        st.session_state.section_key = section_key
                        st.session_state.show_review = True
                        st.session_state.chunks_info = data.get("chunks_info", [])
                        st.session_state.prompt = data.get("prompt", "")
                        st.session_state.section_approved = False
                        st.session_state.edited_section_content = section
                        
                        # Clear previous analysis
                        for key in ['generated_drugs', 'reference_drugs', 'drug_comparison', 'grammar_analysis']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        # Calculate tokens and cost for section generation
                        prompt_content = data.get("prompt", "")
                        input_tokens = count_tokens(prompt_content, st.session_state.selected_model)
                        output_tokens = count_tokens(section, st.session_state.selected_model)
                        cost, cost_breakdown = calculate_cost(input_tokens, output_tokens, st.session_state.selected_model, st.session_state.provider)
                        
                        # Store cost info for display in generation details
                        st.session_state.last_generation_cost = {
                            'model': st.session_state.selected_model,
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'cost': cost
                        }
                        
                        # Update total costs
                        st.session_state.total_tokens_used += input_tokens + output_tokens
                        st.session_state.total_cost += cost
                        
                        # Show cost info
                        show_cost_info(st.session_state.selected_model, input_tokens, output_tokens, cost)
                        
                        st.success("✅ Section generated successfully!")
                        st.info("Navigate to 'Generated Content & Review' to view and edit your section.")
                        
                        # Auto-navigate to review page
                        st.session_state.current_page = "Generated Content & Review"
                        st.rerun()
                    else:
                        st.error("No content was generated. Please try again.")
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")
    
    # Tips
    st.markdown('<div class="section-heading">💡 Tips</div>', unsafe_allow_html=True)
    st.info("📚 Ensure you have uploaded relevant documents in Document Management for better context retrieval.")

def render_content_review():
    """Page 3: Generated Content & Review"""
    st.markdown("""
    <div class="page-header">
        <div class="page-title">📊 Generated Content & Review</div>
        <div class="page-subtitle">Review, edit, and analyze your generated protocol sections</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.get('show_review', False):
        st.info("🚀 No content generated yet. Go to 'Protocol Generator' to create a section first.")
        if st.button("🚀 Go to Generator", type="primary"):
            st.session_state.current_page = "Protocol Generator"
            st.rerun()
        return
    
    # Action Buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✅ Approve Section", disabled=st.session_state.get('section_approved', False)):
            # Get the current content from the editor or the stored content
            current_editor_content = st.session_state.get('content_editor', '')
            current_stored_content = st.session_state.get('edited_section_content', '')
            
            # Use whichever is more recent/available
            content_to_approve = current_editor_content if current_editor_content else current_stored_content
            
            st.session_state.section_approved = True
            st.session_state.approved_section_content = content_to_approve
            
            st.success("🎉 Section approved successfully!")
            st.balloons()
            st.rerun()
    
    with col2:
        if st.button("📤 Export", disabled=not st.session_state.get('section_approved', False)):
            if st.session_state.get('section_approved', False):
                # Show export options
                st.session_state.show_export_modal = True
                st.rerun()
            else:
                st.warning("Please approve the section first")
    
    # Export Modal - Display export options when modal is triggered
    if st.session_state.get('show_export_modal', False):
        st.markdown("---")
        st.markdown("### 📤 Export Options")
        
        # Get content and title for export
        content_to_export = st.session_state.get('approved_section_content', '')
        section_title = st.session_state.get('section_title', 'Protocol_Section')
        
        if content_to_export:
            st.info(f"📄 Ready to export: **{section_title}** ({len(content_to_export):,} characters)")
            
            col_export1, col_export2, col_export3 = st.columns([1, 1, 1])
            
            with col_export1:
                try:
                    # Create Word document
                    word_doc = create_word_doc(content_to_export, section_title)
                    st.download_button(
                        label="📄 Download as Word (.docx)",
                        data=word_doc,
                        file_name=f"{sanitize_filename(section_title)}.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ Word export failed: {str(e)}")
            
            with col_export2:
                try:
                    # Create PDF document
                    pdf_doc = create_pdf(content_to_export, section_title)
                    st.download_button(
                        label="📄 Download as PDF (.pdf)",
                        data=pdf_doc,
                        file_name=f"{sanitize_filename(section_title)}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"❌ PDF export failed: {str(e)}")
            
            with col_export3:
                if st.button("❌ Close Export", key="close_export_modal", use_container_width=True):
                    st.session_state.show_export_modal = False
                    st.rerun()
        else:
            st.error("❌ No approved content available for export")
            st.warning("Please approve a section first before exporting.")
            if st.button("❌ Close", key="close_export_modal_error", use_container_width=True):
                st.session_state.show_export_modal = False
                st.rerun()
        
        st.markdown("---")
    
    with col3:
        if st.button("🔄 Regenerate"):
            # Clear current state and go back to generator
            st.session_state.section_approved = False
            st.session_state.approved_section_content = ""
            st.session_state.current_page = "Protocol Generator"
            st.rerun()
    
    # Main Content Display
    if st.session_state.get('section_approved', False):
        # Show approved content in a styled read-only container
        st.markdown("### ✅ Approved Section Content")
        st.markdown(f"""
        <div style="
            background: #f0fdf4; 
            border: 2px solid #16a34a; 
            border-radius: 8px; 
            padding: 1.5rem; 
            margin: 1rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="
                color: #065f46; 
                font-family: Georgia, serif; 
                line-height: 1.8; 
                font-size: 16px;
                white-space: pre-wrap;
                word-wrap: break-word;
            ">
{st.session_state.get('approved_section_content', '').replace(chr(10), '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Show editable content
        st.markdown("### 📝 Generated Section Content")
        content = st.text_area(
            "Edit section content:",
            value=st.session_state.get('edited_section_content', ''),
            height=400,
            key="content_editor",
            help="Edit the generated content as needed before approval"
        )
        
        # Update stored content when edited
        if content != st.session_state.get('edited_section_content', ''):
            st.session_state.edited_section_content = content
    
    # Analysis Tabs
    st.markdown('<div class="section-heading">📊 Content Analysis</div>', unsafe_allow_html=True)
    
    analysis_tabs = st.tabs(["🧬 Drug Analysis", "📝 Grammar & Consistency", "📊 Generation Details"])
    
    with analysis_tabs[0]:  # Drug Analysis
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Analyze Drugs", type="primary"):
                content_to_analyze = st.session_state.get('content_editor', st.session_state.get('edited_section_content', ''))
                
                with st.spinner("Analyzing drug information..."):
                    try:
                        # Get reference text from the chunks used during generation
                        reference_text = ""
                        if st.session_state.get('chunks_info'):
                            reference_chunks = [chunk.get('content', '') for chunk in st.session_state.chunks_info]
                            reference_text = "\n\n".join(reference_chunks)
                        
                        drug_analysis_req = {
                            "text": content_to_analyze,
                            "reference_text": reference_text,
                            "selected_model": st.session_state.get('selected_model', 'llama-3.3-70b-versatile'),
                            "provider": st.session_state.get('provider', 'groq'),
                            "temperature": st.session_state.get('temperature', 0.2)
                        }
                        
                        response = requests.post("http://localhost:8000/analyze/drugs", json=drug_analysis_req)
                        response.raise_for_status()
                        data = response.json()
                        
                        st.session_state.generated_drugs = data.get("generated_drugs", [])
                        st.session_state.reference_drugs = data.get("reference_drugs", [])
                        st.session_state.drug_comparison = data.get("comparison", {})
                        st.session_state.drug_cost_summary = data.get("cost_summary", {})
                        st.session_state.extraction_method = data.get("extraction_method", "unknown")
                        
                        # Show pure LLM extraction cost info
                        cost_summary = data.get("cost_summary", {})
                        if cost_summary:
                            st.info(f"""
                            💰 **Pure LLM Extraction Cost Summary:**
                            - Method: {data.get('extraction_method', 'Unknown')}
                            - Total Tokens: {cost_summary.get('total_tokens_used', 0):,}
                            - Total Cost: ${cost_summary.get('total_cost_usd', 0):.4f}
                            - Model: {cost_summary.get('model', 'Unknown')}
                            - Approach: {cost_summary.get('extraction_method', 'High accuracy extraction')}
                            """)
                        
                        st.success("✅ Drug analysis completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Drug analysis failed: {str(e)}")
        
        # Display drug analysis results
        if st.session_state.get('generated_drugs') or st.session_state.get('reference_drugs'):
            st.markdown("### 🧬 Drug Analysis Results")
            
            # Show cost summary if available
            cost_summary = st.session_state.get('drug_cost_summary', {})
            extraction_method = st.session_state.get('extraction_method', 'unknown')
            
            if cost_summary:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("#### Analysis Summary")
                with col2:
                    with st.expander("💰 Cost Details", expanded=False):
                        st.markdown(f"""
                        **Extraction Method:** {extraction_method.replace('_', ' ').title()}  
                        **Total Tokens:** {cost_summary.get('total_tokens_used', 0):,}  
                        **Total Cost:** ${cost_summary.get('total_cost_usd', 0):.4f}  
                        **Model:** {cost_summary.get('model', 'Unknown')}  
                        **Approach:** {cost_summary.get('extraction_method', 'High accuracy extraction')}
                        """)
            
            # Summary metrics
            generated_drugs = st.session_state.get('generated_drugs', [])
            reference_drugs = st.session_state.get('reference_drugs', [])
            comparison = st.session_state.get('drug_comparison', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Generated Drugs", len(generated_drugs))
            with col2:
                st.metric("Reference Drugs", len(reference_drugs))
            with col3:
                # Handle both old and new comparison formats
                if 'summary' in comparison:
                    # New LLM-based format
                    perfect_matches = comparison['summary'].get('perfect_matches', 0)
                    partial_matches = comparison['summary'].get('partial_matches', 0)
                    total_matches = perfect_matches + partial_matches
                else:
                    # Old format fallback
                    total_matches = len(comparison.get('matches', []))
                st.metric("Matching Drugs", total_matches)
            with col4:
                if 'summary' in comparison:
                    # New LLM-based format
                    accuracy = comparison['summary'].get('accuracy_percentage', 0)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                elif len(reference_drugs) > 0:
                    # Old format fallback
                    coverage = (total_matches / len(reference_drugs)) * 100
                    st.metric("Coverage", f"{coverage:.1f}%")
                else:
                    st.metric("Coverage", "N/A")
            
            # Detailed comparison tables
            if comparison:
                # Separate tables for better organization
                matches = comparison.get('matches', [])
                missing = comparison.get('missing', [])
                additional = comparison.get('additional', [])
                
                # Table 1: Matched Drugs (Perfect and Partial matches)
                if matches:
                    st.markdown("### 📊 Drug Comparison - Matched Drugs")
                    
                    match_data = []
                    for match in matches:
                        ref_drug = match.get('reference_drug', match.get('reference', {}))
                        gen_drug = match.get('generated_drug', match.get('generated', {}))
                        
                        # Ensure ref_drug and gen_drug are dictionaries, not None
                        if ref_drug is None:
                            ref_drug = {}
                        if gen_drug is None:
                            gen_drug = {}
                        
                        # Handle new LLM-based match types
                        match_type = match.get('match_type', '')
                        if match_type == 'PERFECT_MATCH':
                            status = '✅ Perfect Match'
                        elif match_type == 'PARTIAL_MATCH':
                            status = '🟡 Partial Match'
                        elif match.get('details_match', False):
                            status = '✅ Match'
                        else:
                            status = '❌ No Match'
                        
                        match_data.append({
                            'Drug Name': ref_drug.get('name', 'Unknown'),
                            'Status': status,
                            'Generated Dosage': gen_drug.get('dosage', 'Not specified') or 'Not specified',
                            'Reference Dosage': ref_drug.get('dosage', 'Not specified') or 'Not specified',
                            'Explanation': match.get('explanation', 'No explanation provided')
                        })
                    
                    st.dataframe(
                        match_data, 
                        use_container_width=True,
                        column_config={
                            "Drug Name": st.column_config.TextColumn("Drug Name", width="medium"),
                            "Status": st.column_config.TextColumn("Status", width="medium"),
                            "Generated Dosage": st.column_config.TextColumn("Generated Dosage", width="small"),
                            "Reference Dosage": st.column_config.TextColumn("Reference Dosage", width="small"),
                            "Explanation": st.column_config.TextColumn("Explanation", width="large")
                        },
                        hide_index=True
                    )
                
                # Table 2: Missing Drugs (in reference but not generated)
                if missing:
                    st.markdown("### ❌ Missing Drugs (Found in Reference, Not in Generated)")
                    
                    missing_data = []
                    for missing_item in missing:
                        missing_drug = missing_item.get('drug', missing_item)
                        
                        # Ensure missing_drug is a dictionary, not None
                        if missing_drug is None:
                            missing_drug = {}
                        
                        missing_data.append({
                            'Drug Name': missing_drug.get('name', 'Unknown'),
                            'Reference Dosage': missing_drug.get('dosage', 'Not specified') or 'Not specified',
                            'Reference Source': missing_drug.get('source_context', missing_drug.get('source_text', 'No source'))[:100] + '...' if len(missing_drug.get('source_context', missing_drug.get('source_text', ''))) > 100 else missing_drug.get('source_context', missing_drug.get('source_text', 'No source')),
                            'Explanation': missing_item.get('explanation', 'Drug found in reference but not in generated text')
                        })
                    
                    st.dataframe(
                        missing_data, 
                        use_container_width=True,
                        column_config={
                            "Drug Name": st.column_config.TextColumn("Drug Name", width="medium"),
                            "Reference Dosage": st.column_config.TextColumn("Reference Dosage", width="small"),
                            "Reference Source": st.column_config.TextColumn("Reference Source", width="large"),
                            "Explanation": st.column_config.TextColumn("Explanation", width="large")
                        },
                        hide_index=True
                    )
                
                # Table 3: Additional Drugs (in generated but not reference)
                if additional:
                    st.markdown("### ➕ Additional Drugs (Found in Generated, Not in Reference)")
                    
                    additional_data = []
                    for additional_item in additional:
                        additional_drug = additional_item.get('drug', additional_item)
                        
                        # Ensure additional_drug is a dictionary, not None
                        if additional_drug is None:
                            additional_drug = {}
                        
                        additional_data.append({
                            'Drug Name': additional_drug.get('name', 'Unknown'),
                            'Generated Dosage': additional_drug.get('dosage', 'Not specified') or 'Not specified',
                            'Generated Source': additional_drug.get('source_context', additional_drug.get('source_text', 'No source'))[:100] + '...' if len(additional_drug.get('source_context', additional_drug.get('source_text', ''))) > 100 else additional_drug.get('source_context', additional_drug.get('source_text', 'No source')),
                            'Explanation': additional_item.get('explanation', 'Drug found in generated text but not in reference')
                        })
                    
                    st.dataframe(
                        additional_data, 
                        use_container_width=True,
                        column_config={
                            "Drug Name": st.column_config.TextColumn("Drug Name", width="medium"),
                            "Generated Dosage": st.column_config.TextColumn("Generated Dosage", width="small"),
                            "Generated Source": st.column_config.TextColumn("Generated Source", width="large"),
                            "Explanation": st.column_config.TextColumn("Explanation", width="large")
                        },
                        hide_index=True
                    )
                
                if not matches and not missing and not additional:
                    # Check if we have any drugs at all
                    if len(generated_drugs) > 0 or len(reference_drugs) > 0:
                        st.warning("⚠️ Comparison completed but no clear matches found. This may indicate significant differences between generated and reference content.")
                        
                        # Show what we found
                        if len(generated_drugs) > 0:
                            st.info(f"📋 Generated content contains {len(generated_drugs)} drugs")
                        if len(reference_drugs) > 0:
                            st.info(f"📚 Reference content contains {len(reference_drugs)} drugs")
                    else:
                        st.info("No drugs found for comparison.")
                    
                    # Summary insights
                    st.markdown("### 💡 Analysis Insights")
                    
                    # Handle both old and new comparison formats
                    if 'summary' in comparison:
                        # New LLM-based format
                        summary = comparison['summary']
                        missing_count = summary.get('missing_count', 0)
                        additional_count = summary.get('additional_count', 0)
                        perfect_matches = summary.get('perfect_matches', 0)
                        partial_matches = summary.get('partial_matches', 0)
                        total_matches = perfect_matches + partial_matches
                        total_reference = summary.get('total_reference', len(reference_drugs))
                        accuracy = summary.get('accuracy_percentage', 0)
                    else:
                        # Old format fallback
                        missing_count = len(comparison.get('missing', []))
                        additional_count = len(comparison.get('additional', []))
                        perfect_matches = len([match for match in comparison.get('matches', []) if match.get('details_match', False)])
                        partial_matches = len([match for match in comparison.get('matches', []) if not match.get('details_match', False)])
                        total_matches = len(comparison.get('matches', []))
                        total_reference = len(reference_drugs)
                        accuracy = (perfect_matches / total_reference * 100) if total_reference > 0 else 0
                    
                    # More accurate analysis
                    if missing_count == 0 and additional_count == 0 and perfect_matches == total_matches:
                        st.success("✅ Perfect match! All reference drugs are present with exact details matching.")
                    elif missing_count == 0 and perfect_matches == total_matches:
                        st.success(f"✅ Excellent! All reference drugs match perfectly. Found {additional_count} additional drug(s).")
                    elif missing_count == 0 and perfect_matches > 0:
                        st.warning(f"🟡 Partial match: {perfect_matches}/{total_matches} drugs have perfect details. {partial_matches} have mismatched dosages/details.")
                    elif missing_count == 0 and perfect_matches == 0:
                        st.error(f"❌ Poor match: All drug names found but NO dosages/details match correctly!")
                    elif perfect_matches > 0:
                        st.warning(f"⚠️ Mixed results: {perfect_matches}/{total_reference} perfect matches, {missing_count} missing, {additional_count} additional.")
                    else:
                        st.error(f"❌ Significant issues: {missing_count} missing drugs, {additional_count} additional drugs, {accuracy:.1f}% accuracy.")
                        
                    # Simplified statistics
                    if total_matches > 0 or missing_count > 0:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Perfect Matches", f"{perfect_matches}/{total_reference}")
                        with col2:
                            st.metric("Missing Drugs", missing_count)
                        with col3:
                            st.metric("Accuracy", f"{accuracy:.1f}%")
                # Analysis insights section is shown above when there are matches/missing/additional drugs
            else:
                # Show individual lists if no comparison available
                col1, col2 = st.columns(2)
                
                with col1:
                    if generated_drugs:
                        st.markdown("**🔬 Drugs in Generated Content:**")
                        for drug in generated_drugs:
                            st.markdown(f"• **{drug['name']}**")
                            if drug.get('dosage'):
                                st.markdown(f"  - Dosage: {drug['dosage']}")
                            if drug.get('route'):
                                st.markdown(f"  - Route: {drug['route']}")
                            source_text = drug.get('source_context', drug.get('source_text', ''))
                            if source_text:
                                st.markdown(f"  - Context: *{source_text[:100]}...*" if len(source_text) > 100 else f"  - Context: *{source_text}*")
                
                with col2:
                    if reference_drugs:
                        st.markdown("**📚 Drugs in Reference Documents:**")
                        for drug in reference_drugs:
                            st.markdown(f"• **{drug['name']}**")
                            if drug.get('dosage'):
                                st.markdown(f"  - Dosage: {drug['dosage']}")
                            if drug.get('route'):
                                st.markdown(f"  - Route: {drug['route']}")
                            source_text = drug.get('source_context', drug.get('source_text', ''))
                            if source_text:
                                st.markdown(f"  - Context: *{source_text[:100]}...*" if len(source_text) > 100 else f"  - Context: *{source_text}*")
    
    with analysis_tabs[1]:  # Grammar & Consistency
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("📝 Analyze Grammar", type="primary"):
                content_to_analyze = st.session_state.get('content_editor', st.session_state.get('edited_section_content', ''))
                section_name = st.session_state.get('section_title', 'Generated Section')
                
                with st.spinner("Analyzing grammar and consistency..."):
                    try:
                        analysis_req = {
                            "text": content_to_analyze,
                            "section_name": section_name,
                            "reference_text": ""
                        }
                        
                        response = requests.post("http://localhost:8000/analyze/grammar", json=analysis_req)
                        response.raise_for_status()
                        data = response.json()
                        
                        st.session_state.grammar_analysis = data.get("analysis", {})
                        
                        # Calculate tokens and cost for grammar analysis (local processing, minimal tokens)
                        input_tokens = count_tokens(content_to_analyze, st.session_state.selected_model)
                        output_tokens = 25  # Estimated for analysis results
                        cost, cost_breakdown = calculate_cost(input_tokens, output_tokens, st.session_state.selected_model, st.session_state.provider)
                        
                        # Show cost info
                        show_cost_info(st.session_state.selected_model, input_tokens, output_tokens, cost)
                        
                        st.success("✅ Grammar analysis completed!")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Grammar analysis failed: {str(e)}")
        
        # Display grammar analysis results
        if st.session_state.get('grammar_analysis'):
            analysis = st.session_state.grammar_analysis
            
            # Score display
            col1, col2, col3, col4 = st.columns(4)
            scores = analysis.get('scores', {})
            
            # Standardize ICH compliance field name
            ich_compliance_score = scores.get('ich_compliance', scores.get('compliance', 0))
            
            with col1:
                st.metric("Grammar", f"{scores.get('grammar', 0)}/100", 
                         delta=f"{scores.get('grammar', 0) - 75}" if scores.get('grammar', 0) != 75 else None)
            with col2:
                st.metric("Consistency", f"{scores.get('consistency', 0)}/100",
                         delta=f"{scores.get('consistency', 0) - 75}" if scores.get('consistency', 0) != 75 else None)
            with col3:
                st.metric("ICH Compliance", f"{ich_compliance_score}/100",
                         delta=f"{ich_compliance_score - 80}" if ich_compliance_score != 80 else None)
            with col4:
                st.metric("Overall", f"{scores.get('overall', 0):.1f}/100",
                         delta=f"{scores.get('overall', 0) - 77:.1f}" if scores.get('overall', 0) != 77 else None)
            
            # Issues display
            issues = analysis.get('issues', [])
            
            if issues:
                st.markdown("### 🔍 Issues Found")
                
                # Group issues by category
                issues_by_category = {}
                for issue in issues:
                    category = issue.get('category', issue.get('type', 'Other'))
                    if category not in issues_by_category:
                        issues_by_category[category] = []
                    issues_by_category[category].append(issue)
                
                # Display issues by category
                for category, issue_list in issues_by_category.items():
                    if issue_list:
                        category_name = category.replace('_', ' ').title()
                        
                        # Count issues by severity
                        high_count = len([i for i in issue_list if i.get('severity') == 'high'])
                        medium_count = len([i for i in issue_list if i.get('severity') == 'medium'])
                        low_count = len([i for i in issue_list if i.get('severity') == 'low'])
                        
                        severity_info = ""
                        if high_count > 0:
                            severity_info += f" 🔴{high_count}"
                        if medium_count > 0:
                            severity_info += f" 🟡{medium_count}"
                        if low_count > 0:
                            severity_info += f" 🟢{low_count}"
                        
                        with st.expander(f"{category_name} ({len(issue_list)} issues{severity_info})"):
                            for issue in issue_list:
                                severity = issue.get('severity', 'medium')
                                severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                                
                                message = issue.get('message', 'No message')
                                suggestion = issue.get('suggestion', '')
                                
                                st.markdown(f"{severity_icon} **{message}**")
                                if suggestion:
                                    st.markdown(f"   💡 *{suggestion}*")
                                st.markdown("---")
            
            # ICH M11 Compliance Analysis Section
            st.markdown("### 📋 ICH M11 Compliance Analysis")
            
            # Show ICH compliance score with visual indicator
            if ich_compliance_score >= 80:
                st.success(f"✅ **ICH Compliance Score: {ich_compliance_score}/100** - Good compliance!")
            elif ich_compliance_score >= 60:
                st.warning(f"⚠️ **ICH Compliance Score: {ich_compliance_score}/100** - Moderate compliance. Consider review.")
            else:
                st.error(f"❌ **ICH Compliance Score: {ich_compliance_score}/100** - Poor compliance. Review required.")
            
            # Show ICH-related issues
            ich_issues = [issue for issue in issues if issue.get('type') == 'ich_compliance' or issue.get('category') == 'ich_compliance']
            
            if ich_issues:
                st.markdown("**🔍 ICH M11 Compliance Issues:**")
                
                for issue in ich_issues:
                    severity = issue.get('severity', 'medium')
                    severity_icon = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(severity, '⚪')
                    
                    message = issue.get('message', 'No message')
                    suggestion = issue.get('suggestion', '')
                    
                    st.markdown(f"{severity_icon} **{message}**")
                    if suggestion:
                        st.markdown(f"   💡 *{suggestion}*")
                    st.markdown("---")
            else:
                if ich_compliance_score >= 80:
                    st.info("📋 No specific ICH M11 compliance issues found.")
                elif ich_compliance_score >= 60:
                    st.info("📋 Minor ICH M11 compliance considerations may apply.")
                else:
                    st.warning("📋 Review document for ICH M11 compliance requirements.")
            
            # Show recommendations if available
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                st.markdown("**💡 General Recommendations:**")
                for rec in recommendations:
                    st.markdown(f"• {rec}")
            
            # Show summary if available
            summary = analysis.get('summary', '')
            if summary:
                st.markdown("**📊 Analysis Summary:**")
                st.info(summary)
    
    with analysis_tabs[2]:  # Generation Details
        st.markdown("### 📊 Generation Details")
        
        # Add score interpretation guide
        with st.expander("📈 Understanding Relevance Scores", expanded=False):
            st.markdown("""
            **Score Interpretation:**
            - 🟢 **85%+**: Excellent Match - Highly relevant content
            - 🟡 **75-84%**: Good Match - Well-aligned content  
            - 🟠 **65-74%**: Moderate Match - Reasonably relevant
            - 🔴 **55-64%**: Weak Match - Limited relevance
            - ⚫ **<55%**: Poor Match - Low relevance
            
            *Scores are based on semantic similarity between your query and document content.*
            """)
        
        if st.session_state.get('section_title'):
            st.markdown(f"**Section:** {st.session_state.section_title}")
        
        # Show cost information for the last generation
        if st.session_state.get('last_generation_cost'):
            st.markdown("### 💰 Generation Cost")
            cost_info = st.session_state.last_generation_cost
            show_cost_info(
                cost_info['model'], 
                cost_info['input_tokens'], 
                cost_info['output_tokens'], 
                cost_info['cost']
            )
        
        if st.session_state.get('chunks_info'):
            st.markdown("**Retrieved Context:**")
            for i, chunk in enumerate(st.session_state.chunks_info):
                with st.expander(f"Context {i+1}: {chunk.get('source', 'Unknown')}"):
                    score = chunk.get('score', 0)
                    # Convert score to percentage if it's between 0 and 1
                    if 0 <= score <= 1:
                        score_display = f"{score:.3f} ({score*100:.1f}%)"
                        
                        # Add quality interpretation
                        if score >= 0.85:
                            quality = "🟢 Excellent Match"
                        elif score >= 0.75:
                            quality = "🟡 Good Match"
                        elif score >= 0.65:
                            quality = "🟠 Moderate Match"
                        elif score >= 0.55:
                            quality = "🔴 Weak Match"
                        else:
                            quality = "⚫ Poor Match"
                            
                        st.write(f"**Score:** {score_display} - {quality}")
                    else:
                        score_display = f"{score:.3f}"
                        st.write(f"**Score:** {score_display}")
                    
                    # Show content without truncation
                    content = chunk.get('content', '')
                    if len(content) > 500:
                        st.write(f"**Content:** {content[:500]}...")
                        if st.button(f"📄 Show Full Content", key=f"show_full_{i}"):
                            st.text_area("Full Content:", content, height=200, key=f"full_{i}")
                    else:
                        st.write(f"**Content:** {content}")
        
        if st.session_state.get('prompt'):
            with st.expander("View Full Prompt"):
                st.code(st.session_state.prompt, language="text")

def render_ai_assistant():
    """Page 4: AI Assistant Chatbot"""
    st.markdown("""
    <div class="page-header">
        <div class="page-title">🤖 AI Assistant</div>
        <div class="page-subtitle">Ask questions about clinical trials and protocol development</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Compact Configuration Panel
    with st.expander("⚙️ Configuration", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Search Settings**")
            
            # Initialize search type in session state if not set
            if 'current_search_type' not in st.session_state:
                st.session_state.current_search_type = "documents"
            
            search_type = st.selectbox(
                "Search Mode",
                ["documents", "web", "hybrid"],
                index=["documents", "web", "hybrid"].index(st.session_state.current_search_type),
                help="Documents: Search uploaded documents only\nWeb: Search medical literature online\nHybrid: Search both documents and web",
                key="search_type_radio"
            )
            
            # Update session state when selectbox changes
            if search_type != st.session_state.current_search_type:
                st.session_state.current_search_type = search_type
            
            if search_type == "web" or search_type == "hybrid":
                # Initialize web search type in session state if not set
                if 'current_web_search_type' not in st.session_state:
                    st.session_state.current_web_search_type = "medical"
                
                web_search_type = st.selectbox(
                    "Web Search Type",
                    ["medical", "pubmed", "general"],
                    index=["medical", "pubmed", "general"].index(st.session_state.current_web_search_type),
                    help="Medical: General medical literature\nPubMed: Scientific publications\nGeneral: Broader web search",
                    key="web_search_type_select"
                )
                
                # Update session state when selectbox changes
                if web_search_type != st.session_state.current_web_search_type:
                    st.session_state.current_web_search_type = web_search_type
            else:
                web_search_type = "medical"
                st.session_state.current_web_search_type = web_search_type
            
            # Document collection selection (only for documents and hybrid modes)
            if search_type in ["documents", "hybrid"]:
                st.markdown("**📚 Document Collections**")
                collections = {
                    'ps-index': 'Protocol Summaries',
                    'pt-index': 'Protocol Templates', 
                    'rp-index': 'Reference Protocols',
                    'ib-index': 'Investigator\'s Brochure'
                }
                
                selected_collections = []
                cols = st.columns(2)
                
                for i, (key, name) in enumerate(collections.items()):
                    with cols[i % 2]:
                        if st.checkbox(name, value=True, key=f"collection_{key}_{search_type}"):
                            selected_collections.append(key)
                
                if not selected_collections and search_type in ["documents", "hybrid"]:
                    st.warning("⚠️ Please select at least one document collection.")
                    
                st.session_state.selected_collections = selected_collections
            else:
                st.session_state.selected_collections = []
        
        with col2:
            st.markdown("**🤖 AI Model Settings**")
            
            provider_options = ["groq", "openai"]
            current_provider = st.session_state.get('provider', 'groq')
            provider = st.selectbox("AI Provider", provider_options, index=provider_options.index(current_provider), key="chat_provider")
            
            if provider != st.session_state.provider:
                st.session_state.provider = provider
            
            if provider == "groq":
                model_options = [
                    "llama-3.3-70b-versatile",
                    "llama-3.1-70b-versatile", 
                    "llama-3.1-8b-instant",
                    "mistral-saba-24b"
                ]
            else:  # openai
                model_options = [
                    "gpt-4o",
                    "gpt-4o-mini",
                    "gpt-4-turbo"
                ]
            
            current_model = st.session_state.get('selected_model', model_options[0])
            if current_model not in model_options:
                current_model = model_options[0]
            
            selected_model = st.selectbox("Model", model_options, index=model_options.index(current_model), key="chat_model")
            
            if selected_model != st.session_state.selected_model:
                st.session_state.selected_model = selected_model
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=st.session_state.get('temperature', 0.2),
                    step=0.1,
                    key="chat_temperature"
                )
                
                if temperature != st.session_state.temperature:
                    st.session_state.temperature = temperature
            
            with col2_2:
                chunks_per_collection = st.selectbox(
                    "Chunks per Collection",
                    [1, 2, 3, 4, 5],
                    index=0,  # Default to 1 for optimal performance
                    help="Number of document chunks to retrieve per collection. Lower values = faster responses and better token efficiency.",
                    key="chunks_per_collection"
                )
        
        # Web search setup info (only show if web mode selected)
        if search_type in ["web", "hybrid"]:
            st.info("ℹ️ **Web Search Setup:** Requires SERPER_API_KEY in your .env file. Get a free key from [serper.dev](https://serper.dev)")
    
    st.markdown("---")
    
    # Streamlined Chat Interface
    col1, col2, col3 = st.columns([7, 1.5, 1])
    
    with col1:
        user_query = st.text_input(
            "💬 Ask your question",
            placeholder="e.g., How should I structure the inclusion criteria section?",
            key="chat_input"
        )
    
    with col2:
        send_clicked = st.button("📤 Send", type="primary", use_container_width=True)
    
    with col3:
        if st.button("🗑️ Clear", type="secondary", use_container_width=True):
            st.session_state.chatbot_history = []
            st.rerun()
    
    # Process send button click
    if send_clicked:
            if user_query.strip():
                # Add user message to history
                if 'chatbot_history' not in st.session_state:
                    st.session_state.chatbot_history = []
                
                st.session_state.chatbot_history.append({"role": "user", "content": user_query})
                
                with st.spinner("Thinking..."):
                    try:
                        # Get search type from session state
                        current_search_type = st.session_state.get('current_search_type', 'documents')
                        current_web_search_type = st.session_state.get('current_web_search_type', 'medical')
                        
                        # Prepare collections based on search type
                        collections_to_use = []
                        if current_search_type in ["documents", "hybrid"]:
                            collections_to_use = st.session_state.get('selected_collections', [])
                            if not collections_to_use and current_search_type == "documents":
                                st.error("⚠️ Please select at least one document collection for document search.")
                                return
                        

                        
                        chatbot_req = {
                            "query": user_query,
                            "collections": collections_to_use,
                            "search_type": current_search_type,
                            "provider": st.session_state.provider,
                            "model": st.session_state.selected_model,
                            "web_search_type": current_web_search_type,
                            "chunks_per_collection": chunks_per_collection
                        }
                        

                        
                        response = requests.post("http://localhost:8000/chatbot/chat", json=chatbot_req)
                        response.raise_for_status()
                        data = response.json()
                        
                        ai_response = data.get("answer", "Sorry, I couldn't generate a response.")
                        sources = data.get("sources", [])
                        web_sources = data.get("web_sources", [])
                        search_mode = data.get("search_type", current_search_type)
                        
                        # Format response with improved sources display
                        full_response = ai_response
                        
                        # Add document sources (for documents and hybrid modes)
                        if sources and len(sources) > 0:
                            full_response += "\n\n---\n**📄 Document Sources:**"
                            for i, source in enumerate(sources[:5], 1):  # Show max 5 sources
                                full_response += f"\n{i}. {source}"
                        
                        # Add web sources (for web and hybrid modes)
                        if web_sources and len(web_sources) > 0:
                            if search_mode == "web":
                                full_response += "\n\n---\n**🌐 Web Sources:**"
                            else:
                                full_response += "\n\n**🌐 Web Sources:**"
                            for i, source in enumerate(web_sources[:3], 1):  # Show max 3 web sources
                                title = source.get('title', 'Unknown Title')[:80] + "..." if len(source.get('title', '')) > 80 else source.get('title', 'Unknown Title')
                                url = source.get('link', source.get('url', '#'))
                                domain = url.split('//')[1].split('/')[0] if '//' in url else 'Unknown Source'
                                full_response += f"\n{i}. **[{title}]({url})**"
                                full_response += f"\n   *Source: {domain}*"
                        
                        # Add search mode indicator
                        if search_mode:
                            mode_emoji = {"documents": "📚", "web": "🌐", "hybrid": "🔄"}.get(search_mode, "🔍")
                            full_response += f"\n\n*Search mode: {mode_emoji} {search_mode.title()}*"
                        
                        # Add AI response to history
                        st.session_state.chatbot_history.append({"role": "assistant", "content": full_response})
                        
                        # Calculate tokens and cost
                        input_tokens = count_tokens(user_query, st.session_state.selected_model)
                        output_tokens = count_tokens(ai_response, st.session_state.selected_model)
                        cost, cost_breakdown = calculate_cost(input_tokens, output_tokens, st.session_state.selected_model, st.session_state.provider)
                        
                        # Update total costs
                        st.session_state.total_tokens_used += input_tokens + output_tokens
                        st.session_state.total_cost += cost
                        
                        # Show cost info
                        show_cost_info(st.session_state.selected_model, input_tokens, output_tokens, cost)
                        
                        # Clear input and rerun
                        st.rerun()
                        
                    except requests.exceptions.RequestException as e:
                        # Handle specific HTTP errors
                        error_msg = str(e)
                        if "413" in error_msg or "too large" in error_msg.lower():
                            st.error("⚠️ **Message too long!** Your query or the document content is too large for the selected model. Try:\n\n"
                                   "• Use a shorter, more specific question\n"
                                   "• Select fewer document collections\n" 
                                   "• Switch to OpenAI models (higher token limits)")
                        elif "rate_limit" in error_msg.lower():
                            st.error("⚠️ **Rate limit exceeded!** Please wait a moment before trying again.")
                        else:
                            st.error(f"Network error: {str(e)}")
                    except Exception as e:
                        error_msg = str(e)
                        # Check if it's a user-friendly error message from the backend
                        if "high demand and rate limits" in error_msg:
                            st.error("⚠️ **Service temporarily busy!** I'm experiencing high demand. Please try again in a few minutes or switch to a different AI provider.")
                        elif "too large" in error_msg and "document content" in error_msg:
                            st.error("⚠️ **Content too large!** Try asking a shorter question or selecting fewer document collections.")
                        elif "technical issue" in error_msg:
                            st.error("⚠️ **Technical issue!** Please try again or switch to a different AI provider if the problem persists.")
                        elif "[LLM ERROR]" in error_msg and "413" in error_msg:
                            st.error("⚠️ **Token limit exceeded!** Try asking a shorter, more focused question or switch to OpenAI models.")
                        else:
                            st.error(f"Error: {str(e)}")
    
    # Compact Quick Questions
    with st.expander("❓ Quick Questions", expanded=True):
        quick_questions = [
            "How do I write effective inclusion criteria?",
            "What should be included in the study objectives section?",
            "How do I structure the statistical analysis plan?",
            "What are the key elements of informed consent?",
            "How should I describe the study population?"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_q_{i}", use_container_width=True):
                    # Add question to chat history directly
                    if 'chatbot_history' not in st.session_state:
                        st.session_state.chatbot_history = []
                    
                    st.session_state.chatbot_history.append({"role": "user", "content": question})
                    
                    # Process the question automatically
                    with st.spinner("Thinking..."):
                        try:
                            # Use the same search settings as the current session
                            current_search_type = st.session_state.get('current_search_type', 'documents')
                            current_web_search_type = st.session_state.get('current_web_search_type', 'medical')
                            
                            # Prepare collections based on search type
                            collections_to_use = []
                            if current_search_type in ["documents", "hybrid"]:
                                collections_to_use = st.session_state.get('selected_collections', ["ps-index", "pt-index", "rp-index", "ib-index"])
                            
                            chatbot_req = {
                                "query": question,
                                "collections": collections_to_use,
                                "search_type": current_search_type,
                                "provider": st.session_state.provider,
                                "model": st.session_state.selected_model,
                                "web_search_type": current_web_search_type,
                                "chunks_per_collection": chunks_per_collection
                            }
                            
                            response = requests.post("http://localhost:8000/chatbot/chat", json=chatbot_req)
                            response.raise_for_status()
                            data = response.json()
                            
                            ai_response = data.get("answer", "Sorry, I couldn't generate a response.")
                            sources = data.get("sources", [])
                            web_sources = data.get("web_sources", [])
                            search_mode = data.get("search_type", current_search_type)
                            
                            # Format response with improved sources display (same as manual input)
                            full_response = ai_response
                            
                            # Add document sources (for documents and hybrid modes)
                            if sources and len(sources) > 0:
                                full_response += "\n\n---\n**📄 Document Sources:**"
                                for i, source in enumerate(sources[:5], 1):  # Show max 5 sources
                                    full_response += f"\n{i}. {source}"
                            
                            # Add web sources (for web and hybrid modes)
                            if web_sources and len(web_sources) > 0:
                                if search_mode == "web":
                                    full_response += "\n\n---\n**🌐 Web Sources:**"
                                else:
                                    full_response += "\n\n**🌐 Web Sources:**"
                                for i, source in enumerate(web_sources[:3], 1):  # Show max 3 web sources
                                    title = source.get('title', 'Unknown Title')[:80] + "..." if len(source.get('title', '')) > 80 else source.get('title', 'Unknown Title')
                                    url = source.get('link', source.get('url', '#'))
                                    domain = url.split('//')[1].split('/')[0] if '//' in url else 'Unknown Source'
                                    full_response += f"\n{i}. **[{title}]({url})**"
                                    full_response += f"\n   *Source: {domain}*"
                            
                            # Add search mode indicator
                            if search_mode:
                                mode_emoji = {"documents": "📚", "web": "🌐", "hybrid": "🔄"}.get(search_mode, "🔍")
                                full_response += f"\n\n*Search mode: {mode_emoji} {search_mode.title()}*"
                            
                            # Add AI response to history
                            st.session_state.chatbot_history.append({"role": "assistant", "content": full_response})
                            
                            st.rerun()
                            
                        except requests.exceptions.RequestException as e:
                            # Handle specific HTTP errors
                            error_msg = str(e)
                            if "413" in error_msg or "too large" in error_msg.lower():
                                st.error("⚠️ **Message too long!** Try asking a shorter, more specific question or switch to OpenAI models.")
                            elif "rate_limit" in error_msg.lower():
                                st.error("⚠️ **Rate limit exceeded!** Please wait a moment before trying again.")
                            else:
                                st.error(f"Network error: {str(e)}")
                        except Exception as e:
                            error_msg = str(e)
                            # Check if it's a user-friendly error message from the backend
                            if "high demand and rate limits" in error_msg:
                                st.error("⚠️ **Service temporarily busy!** I'm experiencing high demand. Please try again in a few minutes or switch to a different AI provider.")
                            elif "too large" in error_msg and "document content" in error_msg:
                                st.error("⚠️ **Content too large!** Try asking a shorter question or selecting fewer document collections.")
                            elif "technical issue" in error_msg:
                                st.error("⚠️ **Technical issue!** Please try again or switch to a different AI provider if the problem persists.")
                            elif "[LLM ERROR]" in error_msg and "413" in error_msg:
                                st.error("⚠️ **Token limit exceeded!** Try asking a shorter, more focused question or switch to OpenAI models.")
                            else:
                                st.error(f"Error: {str(e)}")
                            
                    st.rerun()
    
    # Professional Chat History Display
    if st.session_state.get('chatbot_history'):
        # Show latest exchange prominently
        latest_messages = st.session_state.chatbot_history[-2:] if len(st.session_state.chatbot_history) >= 2 else st.session_state.chatbot_history
        
        st.markdown("### 💬 Latest Response")
        for message in latest_messages:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0;">
                    <div style="color: #475569; font-weight: 600; margin-bottom: 0.25rem;">👤 You asked:</div>
                    <div style="color: #1e293b;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: #fefefe; border: 1px solid #d1d5db; border-radius: 8px; padding: 0.75rem; margin: 0.5rem 0; max-height: 400px; overflow-y: auto;">
                    <div style="color: #059669; font-weight: 600; margin-bottom: 0.5rem;">🤖 AI Assistant:</div>
                    <div style="color: #111827; line-height: 1.5;">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Collapsible full history
        if len(st.session_state.chatbot_history) > 2:
            with st.expander(f"📜 View Full Chat History ({len(st.session_state.chatbot_history)//2} conversations)", expanded=False):
                for i in range(len(st.session_state.chatbot_history)-1, -1, -2):  # Show latest first, in pairs
                    if i > 0:  # Ensure we have both user and assistant messages
                        user_msg = st.session_state.chatbot_history[i-1]
                        ai_msg = st.session_state.chatbot_history[i]
                        
                        st.markdown(f"""
                        <div style="border: 1px solid #e5e7eb; border-radius: 6px; margin: 0.75rem 0; padding: 0.5rem;">
                            <div style="background: #f9fafb; padding: 0.5rem; border-radius: 4px; margin-bottom: 0.5rem;">
                                <strong>👤:</strong> {user_msg['content']}
                            </div>
                            <div style="background: #ffffff; padding: 0.5rem; border-radius: 4px; max-height: 200px; overflow-y: auto;">
                                <strong>🤖:</strong> {ai_msg['content'][:500]}{'...' if len(ai_msg['content']) > 500 else ''}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8fafc; border-radius: 8px; margin: 1rem 0;">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">💬</div>
            <div style="color: #64748b;">Start a conversation by asking a question above or using the quick questions below.</div>
        </div>
        """, unsafe_allow_html=True)

def render_settings():
    """Page 5: Settings"""
    st.markdown("""
    <div class="page-header">
        <div class="page-title">⚙️ Settings</div>
        <div class="page-subtitle">Configure AI models and generation parameters</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    st.markdown('<div class="section-heading">🤖 AI Model Configuration</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider_options = ["groq", "openai"]
        current_provider = st.session_state.get('provider', 'groq')
        provider = st.selectbox("AI Provider", provider_options, index=provider_options.index(current_provider))
        
        if provider != st.session_state.provider:
            st.session_state.provider = provider
    
    with col2:
        if provider == "groq":
            model_options = [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile", 
                "llama-3.1-8b-instant",
                "mistral-saba-24b"
            ]
        else:  # openai
            model_options = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo"
            ]
        
        current_model = st.session_state.get('selected_model', model_options[0])
        if current_model not in model_options:
            current_model = model_options[0]
        
        selected_model = st.selectbox("Model", model_options, index=model_options.index(current_model))
        
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
    
    # Generation Parameters
    st.markdown('<div class="section-heading">🎛️ Generation Parameters</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get('temperature', 0.2),
            step=0.1,
            help="Higher values make output more creative but less focused"
        )
        
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
    
    with col2:
        top_k = st.slider(
            "Top-K (Vocabulary)",
            min_value=1,
            max_value=10,
            value=st.session_state.get('top_k', 1),
            step=1,
            help="Number of top tokens to consider for generation"
        )
        
        if top_k != st.session_state.top_k:
            st.session_state.top_k = top_k
    
    # API Configuration
    st.markdown('<div class="section-heading">🔑 API Configuration</div>', unsafe_allow_html=True)
    
    # Display current API key status (masked)
    groq_status = "✅ Configured" if GROQ_API_KEY else "❌ Not configured"
    openai_status = "✅ Configured" if OPENAI_API_KEY else "❌ Not configured"
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Groq API:** {groq_status}")
    with col2:
        st.info(f"**OpenAI API:** {openai_status}")
    
    st.warning("💡 API keys are configured via environment variables (.env file)")
    
    # Save Settings
    if st.button("💾 Save Settings", type="primary"):
        st.success("✅ Settings saved successfully!")
        st.rerun()

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("### 🏥 Clinical Protocol Generator")
        
        # Navigation buttons
        pages = [
            ("📄", "Document Management"),
            ("🚀", "Protocol Generator"),
            ("📊", "Generated Content & Review"),
            ("🤖", "AI Assistant"),
            ("⚙️", "Settings")
        ]
        
        current_page = st.session_state.get('current_page', 'Document Management')
        
        for icon, page_name in pages:
            is_current = (page_name == current_page)
            button_type = "primary" if is_current else "secondary"
            
            if st.button(f"{icon} {page_name}", key=f"nav_{page_name}", 
                        type=button_type, use_container_width=True):
                if page_name != current_page:
                    st.session_state.current_page = page_name
                    st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Generated Sections", 1 if st.session_state.get('show_review') else 0)
        st.metric("Approved Sections", 1 if st.session_state.get('section_approved') else 0)
        
        # Cost tracking
        st.markdown("### 💰 Session Costs")
        total_tokens = st.session_state.get('total_tokens_used', 0)
        total_cost = st.session_state.get('total_cost', 0.0)
        st.metric("Total Tokens", f"{total_tokens:,}")
        if total_cost > 0:
            st.metric("Total Cost", f"${total_cost:.4f}")
        else:
            st.metric("Total Cost", "FREE")

def main():
    """Main application function"""
    get_session_state_defaults()
    render_sidebar()
    
    # Route to appropriate page
    current_page = st.session_state.get('current_page', 'Document Management')
    
    if current_page == "Document Management":
        render_document_management()
    elif current_page == "Protocol Generator":
        render_protocol_generator()
    elif current_page == "Generated Content & Review":
        render_content_review()
    elif current_page == "AI Assistant":
        render_ai_assistant()
    elif current_page == "Settings":
        render_settings()

if __name__ == "__main__":
    main() 