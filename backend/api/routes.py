import sys
from pathlib import Path
import uuid
from datetime import datetime
import json
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import shutil
from backend.core.generator import generate_section_with_user_selection, get_chunk_previews, call_llm
from backend.core.document_loader import build_index
from backend.core.prompt_manager import PromptTemplateManager
from .models import SectionReview, SectionHistory, ReviewResponse
from backend.core.drug_info_extractor import DrugInfoExtractor, DrugInfo
from backend.core.grammar_consistency_analyzer import analyze_protocol_section
from dataclasses import asdict

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prompt template manager
template_path = Path(__file__).parent.parent.parent / "frontend" / "templates" / "prompt_templates.yaml"
prompt_manager = PromptTemplateManager(template_path)

# In-memory storage for section reviews (replace with database in production)
section_reviews = {}

class SectionRequest(BaseModel):
    user_prompt: str
    selected_chunks: Dict[str, List[str]]
    section_key: str
    selected_model: str
    provider: str
    temperature: float
    top_k: int

# IndexRequest removed - no longer needed

class AnalysisRequest(BaseModel):
    text: str
    reference_text: Optional[str] = None # Include reference text for comparison if needed
    selected_model: str
    provider: str
    temperature: float

@app.get("/available_section_templates")
async def get_available_section_templates():
    """Get available section templates."""
    try:
        templates = prompt_manager.list_templates()
        return templates
    except Exception as e:
        logging.error(f"Error getting templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_section_with_chunks")
async def generate_section_with_chunks(request: SectionRequest):
    """Generate a section with selected chunks."""
    try:
        # Get the template for the selected section
        template = prompt_manager.get_template(request.section_key)
        if not template:
            raise HTTPException(status_code=400, detail=f"No template found for section {request.section_key}")
            
        # Combine template context with user prompt
        full_prompt = f"{template['context']}\n\n{template['input_instructions']}\n\n{template['output_template']}\n\n{template['other_instructions']}\n\nAdditional Instructions: {request.user_prompt}"
        
        output, prompt, chunks_info, model, provider = generate_section_with_user_selection(
            full_prompt,
            request.selected_chunks,
            top_k=request.top_k,
            model=request.selected_model,
            provider=request.provider,
            temperature=request.temperature
        )
        
        if not output or output.startswith("[ERROR]"):
             raise HTTPException(status_code=500, detail=output if output else "Failed to generate section content")
            
        # Create an initial review entry in the backend storage
        if request.section_key not in section_reviews:
             section_reviews[request.section_key] = SectionHistory(
                 section_key=request.section_key,
                 versions=[],
                 current_version=0
             )
             
        initial_review = SectionReview(
            section_key=request.section_key,
            original_content=output,
            edited_content=output, # Initially edited content is same as original
            status="generated",
            reviewer_comments=None,
            timestamp=datetime.now(),
            version=section_reviews[request.section_key].current_version + 1
        )
        section_reviews[request.section_key].versions.append(initial_review)
        section_reviews[request.section_key].current_version += 1

        return {
            "output": output,
            "prompt": prompt,
            "chunks_info": chunks_info,
            "model": model,
            "provider": provider
        }
    except Exception as e:
        logging.error(f"Error generating section: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint removed - replaced with upload_and_index

@app.post("/upload_and_index")
async def upload_and_index_documents(
    files: List[UploadFile] = File(...),
    collection_type: str = Form(...)
):
    """Upload files and index them into specified collection"""
    try:
        logging.info(f"Upload request: collection_type={collection_type}, files={[f.filename for f in files]}")
        
        # Map collection types to folder paths and collection names
        collection_mapping = {
            'PS': {'folder': 'data/uploads/protocol_summaries', 'collection': 'ps-index'},
            'PT': {'folder': 'data/uploads/protocol_templates', 'collection': 'pt-index'},
            'RP': {'folder': 'data/uploads/reference_protocols', 'collection': 'rp-index'},
            'IB': {'folder': 'data/uploads/investigators_brochure', 'collection': 'ib-index'}
        }
        
        if collection_type not in collection_mapping:
            logging.error(f"Invalid collection type: {collection_type}")
            raise HTTPException(status_code=400, detail="Invalid collection type")
        
        config = collection_mapping[collection_type]
        folder_path = config['folder']
        collection_name = config['collection']
        logging.info(f"Using folder: {folder_path}, collection: {collection_name}")
        
        # Create upload directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        logging.info(f"Directory created/verified: {folder_path}")
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file.filename and file.filename.endswith(('.pdf', '.docx')):
                file_path = os.path.join(folder_path, file.filename)
                logging.info(f"Saving file: {file.filename} to {file_path}")
                with open(file_path, "wb") as buffer:
                    content = await file.read()
                    buffer.write(content)
                saved_files.append(file.filename)
                logging.info(f"File saved successfully: {file.filename}")
            else:
                logging.warning(f"Skipping invalid file: {file.filename}")
        
        logging.info(f"Total files saved: {len(saved_files)}")
        
        # Index the documents
        if saved_files:
            logging.info(f"Starting indexing for {len(saved_files)} files")
            index = build_index(folder_path, collection_name)
            if index is None:
                logging.error("build_index returned None")
                raise HTTPException(status_code=500, detail="Failed to build index")
            
            logging.info("Indexing completed successfully")
            success_response = {
                "status": "success", 
                "message": f"Uploaded and indexed {len(saved_files)} files to {collection_type}",
                "files": saved_files
            }
            logging.info(f"Returning success response: {success_response}")
            return success_response
        else:
            logging.error("No valid files were saved")
            raise HTTPException(status_code=400, detail="No valid files uploaded")
            
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logging.error(f"Unexpected error uploading and indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/section/review", response_model=ReviewResponse)
async def submit_section_review(review: SectionReview):
    """Submit a section for review"""
    try:
        review_id = str(uuid.uuid4())
        review.timestamp = datetime.now()
        
        if review.section_key not in section_reviews:
            section_reviews[review.section_key] = SectionHistory(
                section_key=review.section_key,
                versions=[],
                current_version=0
            )
        
        section_reviews[review.section_key].versions.append(review)
        section_reviews[review.section_key].current_version += 1
        
        return ReviewResponse(
            success=True,
            message="Section submitted for review successfully",
            review_id=review_id
        )
    except Exception as e:
        logging.error(f"Error submitting section review: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/section/history/{section_key}", response_model=SectionHistory)
async def get_section_history(section_key: str):
    """Get section review history"""
    if section_key not in section_reviews:
        raise HTTPException(status_code=404, detail="Section not found")
    return section_reviews[section_key]

@app.post("/section/approve", response_model=ReviewResponse)
async def approve_section(review: SectionReview):
    """Approve a section"""
    try:
        if review.section_key not in section_reviews:
            raise HTTPException(status_code=404, detail="Section not found")
        
        review.status = "approved"
        review.timestamp = datetime.now()
        section_reviews[review.section_key].versions.append(review)
        section_reviews[review.section_key].current_version += 1
        
        return ReviewResponse(
            success=True,
            message="Section approved successfully",
            review_id=str(uuid.uuid4())
        )
    except Exception as e:
        logging.error(f"Error approving section: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/drugs")
async def analyze_drug_info(request: AnalysisRequest):
    """Analyze drug information comparing generated content with reference documents using pure LLM."""
    try:
        # Import the LLM-based extractor
        from backend.core.llm_drug_extractor import LLMDrugExtractor
        
        # Initialize the LLM-based drug extractor
        llm_extractor = LLMDrugExtractor(
            model=request.selected_model,
            provider=request.provider
        )
        
        # Use LLM-based analysis if reference text is provided
        if request.reference_text:
            analysis_result = llm_extractor.analyze_drug_information(
                generated_text=request.text,
                reference_text=request.reference_text
            )
            
            return {
                "generated_drugs": analysis_result["generated_drugs"],
                "reference_drugs": analysis_result["reference_drugs"],
                "comparison": analysis_result["comparison"],
                "extraction_method": analysis_result["extraction_method"],
                "model_used": analysis_result["model_used"],
                "cost_summary": analysis_result["cost_summary"],
                "total_generated": len(analysis_result["generated_drugs"]),
                "total_reference": len(analysis_result["reference_drugs"])
            }
        else:
            # Fallback to just extracting from generated text
            generated_drugs = llm_extractor.extract_drugs_from_text(request.text, "generated")
            cost_summary = llm_extractor.get_cost_summary()
            
            return {
                "generated_drugs": [asdict(drug) for drug in generated_drugs],
                "reference_drugs": [],
                "comparison": None,
                "extraction_method": "pure_llm",
                "model_used": request.selected_model,
                "cost_summary": cost_summary,
                "total_generated": len(generated_drugs),
                "total_reference": 0
            }
            
    except Exception as e:
        logging.error(f"Error analyzing drug information: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class GrammarAnalysisRequest(BaseModel):
    text: str
    section_name: Optional[str] = ""
    reference_text: Optional[str] = ""

@app.post("/analyze/grammar")
async def analyze_grammar_consistency(request: GrammarAnalysisRequest):
    """Analyze grammar, consistency, and ICH M11 compliance for protocol sections."""
    try:
        logging.info(f"Starting grammar analysis for section: {request.section_name}")
        
        # Use the existing analyzer function
        analysis_result = analyze_protocol_section(
            text=request.text,
            section_name=request.section_name,
            reference_text=request.reference_text or ""
        )
        
        return {
            "success": True,
            "analysis": analysis_result,
            "message": "Analysis completed successfully"
        }
        
    except Exception as e:
        logging.error(f"Error in grammar analysis: {e}")
        raise HTTPException(
            status_code=500, 
            detail={
                "success": False,
                "message": f"Analysis failed: {str(e)}",
                "analysis": {
                    "scores": {"grammar": 0, "consistency": 0, "compliance": 0, "overall": 0},
                    "issues": [],
                    "summary": "Analysis could not be completed",
                    "recommendations": ["Please check the input and try again"],
                    "total_issues": 0
                }
            }
        )

# Chatbot API endpoints
class ChatRequest(BaseModel):
    query: str
    collections: List[str]
    search_type: str = "documents"  # "documents" or "web"
    provider: str = "openai"
    model: str = "gpt-3.5-turbo"
    web_search_type: str = "medical"  # "medical", "pubmed", or "general"
    chunks_per_collection: int = 1  # Number of chunks per collection (default: 1 for optimal performance)

@app.post("/chatbot/chat")
async def chat_with_assistant(request: ChatRequest):
    """Chat with documents and/or web search"""
    try:
        # Import from the new chatbot module
        from backend.core.chatbot import chat_with_documents_and_web
        
        # Use the unified chatbot function that handles all search types
        response = chat_with_documents_and_web(
            query=request.query,
            collections=request.collections,
            search_type=request.search_type,
            provider=request.provider,
            model=request.model,
            web_search_type=request.web_search_type,
            chunks_per_collection=request.chunks_per_collection
        )
        
        return {
            "success": True,
            "answer": response["answer"],
            "sources": response.get("sources", []),
            "web_sources": response.get("web_sources", []),
            "search_type": response.get("search_type", request.search_type)
        }
        
    except Exception as e:
        logging.error(f"Error in chatbot: {e}")
        raise HTTPException(status_code=500, detail=str(e))
