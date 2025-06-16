from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class SectionRequest(BaseModel):
    section: str
    subsection: str
    drug_name: str
    phase: str
    user_input: str

class SectionRequestWithChunks(BaseModel):
    user_prompt: str
    selected_chunks: Dict[str, List[str]]
    section_key: str
    selected_model: str
    provider: str
    temperature: float
    top_k: int 

class SectionReview(BaseModel):
    section_key: str
    original_content: str
    edited_content: Optional[str]
    status: str  # "generated", "under_review", "approved", "rejected"
    reviewer_comments: Optional[str]
    timestamp: datetime
    version: int

class SectionHistory(BaseModel):
    section_key: str
    versions: List[SectionReview]
    current_version: int

class ReviewResponse(BaseModel):
    success: bool
    message: str
    review_id: Optional[str] 