# Clinical Trial Protocol Generator - Complete Project Documentation

## Project Overview
This is a comprehensive AI-powered Clinical Trial Protocol Generator designed for researchers, medical writers, and pharmaceutical professionals. The system combines advanced AI language models, vector databases, and sophisticated analysis tools to generate ICH M11-compliant clinical trial protocols based on uploaded reference documents.

## System Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend       │    │   External      │
│   (Streamlit)   │◄──►│   (FastAPI)      │◄──►│   Services      │
│                 │    │                  │    │                 │
│ • Multi-page UI │    │ • REST API       │    │ • Qdrant Cloud  │
│ • Real-time UI  │    │ • Document Proc. │    │ • OpenAI API    │
│ • Cost Tracking │    │ • LLM Integration│    │ • Groq API      │
│ • Chat Interface│    │ • Analysis Tools │    │ • Serper API    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Frontend Architecture (`frontend/app.py` - 1,705 lines)

#### Core Components:
1. **Multi-Page Application Structure**:
   - **Document Management**: Upload and organize reference materials
   - **Protocol Generator**: AI-powered section generation
   - **Content Review**: Edit, analyze, and approve content
   - **AI Assistant**: Interactive chatbot with document/web search
   - **Settings**: Model configuration and preferences

2. **Advanced UI Features**:
   - **Modern CSS Styling**: Custom gradient themes and responsive design
   - **Real-time Cost Tracking**: Token usage and API cost monitoring
   - **Interactive Sidebar**: Expandable document upload sections
   - **Tabbed Interface**: Organized content presentation
   - **Export Modal**: Word/PDF generation with progress tracking

3. **State Management**:
   - **Session State**: Persistent user data across page refreshes
   - **Template Caching**: ICH M11 section templates cached for performance
   - **Cost Accumulation**: Running total of API usage and costs
   - **Chat History**: Conversation memory for AI assistant

### Backend Architecture

#### API Layer (`backend/api/routes.py` - 388 lines)
```python
# Core Endpoints
POST /upload_and_index           # Document upload and vector indexing
GET /available_section_templates # ICH M11 template retrieval
POST /generate_section_with_chunks # AI section generation
POST /chatbot/chat              # Interactive assistant

# Analysis Endpoints (AI-Free)
POST /analyze/drugs             # Drug information extraction
POST /analyze/grammar           # Grammar and ICH compliance checking

# Review Workflow
POST /section/review            # Section review submission
POST /section/approve           # Section approval
GET /section/history/{key}      # Version history
```

#### Core Processing Layer (`backend/core/`)

**1. Document Processing (`document_loader.py` - 280 lines)**
```python
# Document Processing Pipeline
def build_index(folder_path: str, collection_name: str) -> VectorStoreIndex:
    # 1. Extract text from PDF/DOCX files
    # 2. Create Document objects with metadata
    # 3. Chunk text (1024 tokens, 100 overlap)
    # 4. Generate OpenAI embeddings
    # 5. Store in Qdrant vector database
    # 6. Return queryable index
```

**2. LLM Integration (`generator.py` - 704 lines)**
```python
# Multi-Provider LLM Support
def call_llm(prompt, model, provider, max_tokens=800, temperature=0.2):
    # Providers: OpenAI (GPT-4 variants), Groq (Llama 3.x, Gemma 2)
    # Features: Token counting, rate limit handling, automatic fallbacks
    # Error handling: Graceful degradation with alternative models
```

**3. Drug Information Extraction (`drug_info_extractor.py` - 655 lines)**
```python
class DrugInfoExtractor:
    # AI-Free drug extraction using:
    # - Comprehensive drug database (50+ known drugs)
    # - Pattern recognition (suffix patterns: -pril, -sartan, -statin)
    # - MedSpaCy NLP integration
    # - Dosage, frequency, route, and form extraction
    # - Reference comparison and coverage analysis
```

**4. Grammar Analysis (`grammar_consistency_analyzer.py` - 913 lines)**
```python
class GrammarConsistencyAnalyzer:
    # Comprehensive medical writing analysis:
    # - LanguageTool integration with medical term filtering
    # - ICH M11 compliance validation
    # - Medical terminology consistency checking
    # - Scoring system (0-100 for grammar, consistency, compliance)
```

**5. AI Assistant (`chatbot.py` - 359 lines)**
```python
# Interactive assistant with:
# - Document search capabilities
# - Web search integration (Serper API)
# - Source citation and attribution
# - Conversation history management
```

**6. Configuration Management (`config.py` - 67 lines)**
```python
# Environment-based configuration
def get_config():
    return {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'QDRANT_URL': os.getenv('QDRANT_URL'),
        'EMBEDDING_MODEL': 'text-embedding-3-small',
        # ... 20+ configuration parameters
    }
```

#### Utility Layer (`backend/utils/`)

**Token Management (`token_utils.py` - 202 lines)**
```python
# Cost calculation and optimization
def count_tokens(text: str, model: str) -> int:
    # Accurate token counting for different models
    # Cost calculation per API call
    # Budget tracking and warnings
```

**Export Functionality (`export_utils.py` - 65 lines)**
```python
# Document generation
def create_word_doc(content: str, title: str) -> BytesIO:
    # Professional Word document generation
    # PDF export with ReportLab
    # Filename sanitization
```

## Core Features Deep Dive

### 1. Document Processing & Vector Search

#### Upload Pipeline:
```
File Upload (PDF/DOCX) → Text Extraction → Metadata Extraction → 
Document Chunking (1024 tokens) → OpenAI Embedding Generation → 
Qdrant Vector Storage → Index Building → Search Capability
```

#### Vector Database Collections:
- **`ps-index`**: Protocol Summaries (high-level study descriptions)
- **`pt-index`**: Protocol Templates (ICH M11 structural templates)
- **`rp-index`**: Reference Protocols (previously approved protocols)
- **`ib-index`**: Investigator's Brochures (drug safety/efficacy data)

#### Intelligent Retrieval:
```python
def get_top_k_chunks_with_scores(collection: str, query: str, top_k: int = 2):
    # Semantic similarity search using cosine distance
    # Returns: chunk_id, chunk_text, relevance_score
    # Metadata includes source document and section information
```

### 2. AI-Powered Section Generation

#### Template System:
- **ICH M11 Compliant**: 15+ protocol section templates
- **YAML Configuration**: Structured template definitions
- **Dynamic Prompts**: Context-aware prompt generation
- **User Customization**: Additional instruction support

#### Generation Process:
```python
def generate_section_with_user_selection(
    full_prompt: str,
    selected_chunks: Dict[str, List[str]],
    top_k: int = 2,
    model: str = "gpt-4-turbo",
    provider: str = "openai",
    temperature: float = 0.2
):
    # 1. Combine template with user instructions
    # 2. Retrieve relevant document chunks
    # 3. Build context-aware prompt
    # 4. Call LLM with fallback handling
    # 5. Return generated content with metadata
```

#### Multi-LLM Support:
- **OpenAI Models**: GPT-4, GPT-4-Turbo, GPT-4o, GPT-4o-mini
- **Groq Models**: Llama-3.3-70B, Llama-3.1-8B, Gemma2-9B
- **Fallback Logic**: Automatic model switching on rate limits
- **Cost Optimization**: Token limits and budget controls

### 3. Advanced Analysis Systems (AI-Free)

#### Drug Information Analysis:
```python
# Extraction Methods:
# 1. Regex-based pattern matching (primary)
# 2. MedSpaCy NLP processing (alternative)
# 3. Hybrid approach (maximum accuracy)

Known Drug Database:
- Diabetes: semaglutide, metformin, sitagliptin, etc.
- Cardiovascular: atorvastatin, lisinopril, losartan, etc.
- Antibiotics: amoxicillin, azithromycin, ciprofloxacin, etc.
- Pain/Inflammation: ibuprofen, naproxen, celecoxib, etc.

Information Extracted:
- Drug name, dosage, frequency, route, form
- Source text context
- Comparison with reference documents
- Coverage analysis and recommendations
```

#### Grammar & Consistency Analysis:
```python
# Analysis Components:
# 1. Grammar checking (LanguageTool with medical filtering)
# 2. Medical terminology consistency
# 3. ICH M11 compliance validation
# 4. Medical writing style assessment

Scoring System:
- Grammar Score (0-100): Based on filtered grammar issues
- Consistency Score (0-100): Terminology and style consistency
- Compliance Score (0-100): ICH M11 regulatory compliance
- Overall Score: Weighted average of component scores
```

### 4. Interactive AI Assistant

#### Chatbot Capabilities:
- **Document Search**: Query indexed protocol documents
- **Web Search**: External medical literature search (Serper API)
- **Hybrid Search**: Combined document and web results
- **Source Attribution**: Proper citation of sources
- **Conversation Memory**: Persistent chat history

#### Search Integration:
```python
def chat_with_documents_and_web(
    query: str,
    collections: List[str],
    search_type: str = "documents",  # "documents", "web", "both"
    provider: str = "openai",
    model: str = "gpt-3.5-turbo",
    web_search_type: str = "medical"  # "medical", "pubmed", "general"
):
    # Intelligent search routing and result synthesis
```

## Technology Stack

### Core Technologies
- **Backend Framework**: FastAPI 0.104.0+
- **Frontend Framework**: Streamlit 1.28.0+
- **Vector Database**: Qdrant Cloud
- **Document Processing**: PyPDF2 3.0.0+, python-docx 0.8.11+
- **AI Integration**: OpenAI 1.0.0+, Groq 0.3.0+
- **Embeddings**: OpenAI text-embedding-3-small

### Specialized Libraries
- **Vector Operations**: LlamaIndex 0.10.24
- **Medical NLP**: MedSpaCy, spaCy
- **Grammar Checking**: language-tool-python 2.7.1+
- **Text Analysis**: PyYAML 6.0+, regex 2023.10.3+
- **Export**: ReportLab 4.0.0+
- **HTTP Client**: requests 2.31.0+

### External Services
- **Qdrant Cloud**: Vector database hosting
- **OpenAI API**: Premium LLM models and embeddings
- **Groq API**: Free high-performance LLM models
- **Serper API**: Web search functionality

## Performance & Optimization

### Caching Strategy
```python
# Multi-level caching for optimal performance
_engine_cache = {}  # Vector index caching
session_state = {}  # UI state persistence
template_cache = {} # ICH M11 template caching
```

### Error Handling & Resilience
```python
# Comprehensive error handling
try:
    # Primary model attempt
    response = call_llm(prompt, model, provider)
except RateLimitError:
    # Automatic fallback to alternative models
    response = fallback_model_call(prompt)
except Exception as e:
    # Graceful degradation with user feedback
    return error_response_with_guidance(e)
```

### Token Optimization
```python
# Intelligent token management
def check_prompt_size_and_truncate(prompt: str, provider: str, model: str):
    # Model-specific token limits
    # Smart truncation preserving important context
    # Cost-aware generation
```

## Setup and Installation

### Prerequisites
- **Python**: 3.8 or higher
- **API Keys**: OpenAI, Groq (optional), Qdrant Cloud, Serper (optional)
- **System Requirements**: 4GB RAM minimum, 8GB recommended

### Environment Configuration
```env
# Core Configuration
OPENAI_API_KEY=sk-your-openai-key
GROQ_API_KEY=gsk_your-groq-key
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key
EMBEDDING_MODEL=text-embedding-3-small

# Optional Services
SERPER_API_KEY=your-serper-key  # For web search
DEBUG=true  # Development mode

# Model Configuration
DEFAULT_MODEL=gpt-4-turbo
DEFAULT_PROVIDER=openai
DEFAULT_TEMPERATURE=0.2
```

### Installation Process
```bash
# 1. Repository Setup
git clone <repository-url>
cd clinical-trial-protocol-generator

# 2. Virtual Environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 3. Dependencies
pip install -r requirements.txt

# 4. Environment Setup
cp .env.example .env
# Edit .env with your API keys

# 5. Application Launch
# Terminal 1: Backend
uvicorn backend.api.routes:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
streamlit run frontend/app.py

# 6. Access
# Frontend: http://localhost:8501
# API Docs: http://localhost:8000/docs
```

## Usage Workflow

### Document Management Workflow
```
1. Document Upload
   ├── Select document type (PS/PT/RP/IB)
   ├── Upload PDF/DOCX files
   ├── Automatic text extraction
   ├── Vector embedding generation
   └── Qdrant indexing

2. Document Processing
   ├── Chunking (1024 tokens, 100 overlap)
   ├── Metadata extraction (filename, section)
   ├── Vector storage with cosine similarity
   └── Search index creation
```

### Protocol Generation Workflow
```
1. Section Selection
   ├── Choose ICH M11 template
   ├── Add custom instructions (optional)
   └── Configure model parameters

2. Context Retrieval
   ├── Semantic search across documents
   ├── Relevance scoring and ranking
   └── Chunk selection and preview

3. AI Generation
   ├── Prompt composition with context
   ├── LLM processing with fallbacks
   ├── Content generation and validation
   └── Source attribution

4. Review & Analysis
   ├── Content editing interface
   ├── Drug information analysis
   ├── Grammar and compliance checking
   └── Export to Word/PDF
```

### AI Assistant Workflow
```
1. Query Processing
   ├── Natural language understanding
   ├── Search type determination
   └── Collection selection

2. Information Retrieval
   ├── Document search (vector similarity)
   ├── Web search (Serper API)
   └── Result synthesis

3. Response Generation
   ├── Context-aware AI response
   ├── Source citation and attribution
   └── Conversation history update
```

## Quality Assurance & Validation

### Testing Strategy
- **Unit Tests**: Core functionality validation
- **Integration Tests**: API endpoint testing
- **Performance Tests**: Load and stress testing
- **Medical Accuracy**: Clinical expert review

### Compliance & Standards
- **ICH M11**: International harmonization guidelines
- **Good Clinical Practice (GCP)**: Regulatory compliance
- **Data Privacy**: Local storage, secure API handling
- **Medical Terminology**: Standardized vocabulary usage

## Security & Privacy

### Data Protection
- **Local Storage**: All documents stored locally
- **API Security**: Environment variable API key management
- **Session Isolation**: User data separated per session
- **No Data Persistence**: No permanent user data storage

### API Security
- **Rate Limiting**: Built-in request throttling
- **Error Handling**: Secure error messages
- **Input Validation**: Pydantic model validation
- **CORS Configuration**: Controlled cross-origin access

## Future Enhancements

### Planned Features
- **User Authentication**: Multi-user support with role-based access
- **Database Backend**: Persistent storage for protocols and history
- **Advanced Templates**: Additional ICH sections and custom templates
- **Batch Processing**: Multiple section generation
- **API Integration**: REST API for external systems

### Technical Improvements
- **Async Processing**: Improved performance for large documents
- **Advanced Search**: Hybrid vector + keyword search
- **Mobile Optimization**: Responsive design for mobile devices
- **Monitoring**: Application performance monitoring
- **Automated Testing**: Comprehensive test suite

## Contributing

### Development Guidelines
- **Code Style**: PEP 8 compliance
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit tests for new features
- **Version Control**: Clear commit messages

### Architecture Principles
- **Modularity**: Clear separation of concerns
- **Scalability**: Horizontal scaling capability
- **Maintainability**: Clean, documented code
- **Performance**: Efficient resource utilization

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Support & Contact
For technical support, feature requests, or contributions, please refer to the project repository issues page or contact the development team.
