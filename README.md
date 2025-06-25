# Clinical Trial Protocol Generator 🏥

> **An AI-powered platform for generating ICH M11-compliant clinical trial protocols with comprehensive analysis tools**

## ✨ What is this?

The Clinical Trial Protocol Generator is a comprehensive, AI-powered platform designed for medical writers, researchers, and pharmaceutical professionals. It combines advanced language models with specialized medical analysis tools to generate high-quality clinical trial protocols based on your reference documents.

## 🚀 Key Features

### 📁 **Smart Document Management**
- **Multi-format Support**: Upload PDF and DOCX files
- **Organized Categories**: Protocol Summaries (PS), Protocol Templates (PT), Reference Protocols (RP), Investigator's Brochures (IB)
- **Automatic Processing**: Instant text extraction and AI-powered indexing
- **Vector Search**: Semantic search across all your documents

### 🤖 **AI-Powered Protocol Generation**
- **Multiple AI Models**: OpenAI GPT-4 family and free Groq models (Llama 3.x, Gemma 2)
- **ICH M11 Compliance**: 15+ pre-built section templates following international standards
- **Context-Aware**: Automatically finds relevant information from your uploaded documents
- **Customizable**: Add your own instructions to tailor the output

### 🔍 **Advanced Analysis Tools **
- **Drug Information Extraction**: Automatically identify drugs, dosages, frequencies, and administration routes
- **Grammar & Consistency Checking**: Medical writing quality assessment with ICH M11 compliance validation
- **Reference Comparison**: Compare generated content against your source documents
- **Cost-Free Analysis**: No API costs for analysis tools

### 💬 **Intelligent Medical Writer Assistant**
- **Document Chat**: Ask questions about your uploaded documents
- **Web Search**: Search external medical literature and PubMed
- **Source Attribution**: Get properly cited responses with document references
- **Quick Templates**: Pre-built queries for common medical writing tasks

### 📊 **Professional Features**
- **Real-time Cost Tracking**: Monitor API usage and costs
- **Export Options**: Generate Word documents and PDFs
- **Version Control**: Track changes and approvals
- **Modern UI**: Professional, responsive interface

## 🎯 Who is this for?

- **Medical Writers**: Generate protocol sections faster with AI assistance
- **Clinical Researchers**: Create compliant protocols based on existing templates
- **Pharmaceutical Companies**: Streamline protocol development with standardized templates
- **Regulatory Affairs**: Ensure ICH M11 compliance and consistency
- **Academic Researchers**: Access to free AI models for budget-conscious projects

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Python) - Modern, responsive web interface
- **Backend**: FastAPI - High-performance API with automatic documentation
- **AI Models**: OpenAI GPT-4 family, Groq Llama 3.x/Gemma 2
- **Vector Database**: Qdrant Cloud - Semantic search and document retrieval
- **Analysis**: MedSpaCy, LanguageTool - Medical NLP and grammar checking
- **Export**: ReportLab, python-docx - Professional document generation

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API keys for AI services (OpenAI required, Groq optional for free models)
- Qdrant Cloud account (free tier available)

### Installation

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd clinical-trial-protocol-generator
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or venv\Scripts\activate  # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Launch Application**
   ```bash
   # Terminal 1: Start Backend
   uvicorn backend.api.routes:app --reload

   # Terminal 2: Start Frontend  
   streamlit run frontend/app.py
   ```

5. **Access the Application**
   - **Main Interface**: http://localhost:8501
   - **API Documentation**: http://localhost:8000/docs

### Local Redis Stack (for development)

To run Redis Stack locally (recommended for development):

```bash
docker run -d --name redis-stack-server -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```

- This will start Redis Stack on port 6379 (the default Redis port).
- You can access the RedisInsight UI at [http://localhost:8001](http://localhost:8001) for visual management and monitoring.

Set your `.env`:

```
REDIS_URL=redis://localhost:6379/0
```

## 📖 How to Use

### 1. Upload Your Documents
- Use the sidebar to upload your reference materials
- Organize by type: Protocol Summaries, Templates, Reference Protocols, Investigator's Brochures
- Files are automatically processed and indexed for AI search

### 2. Generate Protocol Sections
- Select an ICH M11-compliant section template
- Add any specific instructions or requirements
- Choose your AI model (OpenAI for premium, Groq for free)
- Generate AI-powered content with source citations

### 3. Review and Analyze
- Edit the generated content as needed
- Run drug information analysis to extract and validate drug mentions
- Perform grammar and compliance checking
- Compare against reference documents

### 4. Chat with Your Documents
- Use the AI assistant to ask questions about your documents
- Search external medical literature
- Get properly cited responses with source links

### 5. Export and Share
- Export to professional Word documents
- Generate PDF reports
- Track costs and usage

## 💡 Example Use Cases

### Medical Writer
*"I need to write a Background and Rationale section for a diabetes study"*
- Upload investigator's brochures and reference protocols
- Select "Background and Rationale" template
- Add specific study details
- Generate AI-powered content with proper citations
- Validate drug information and check compliance

### Clinical Researcher
*"I want to create a new protocol based on previous studies"*
- Upload existing protocols and templates
- Generate multiple sections using different templates
- Compare generated content against references
- Ensure ICH M11 compliance before submission

### Regulatory Affairs Professional
*"I need to check if our protocol meets ICH M11 standards"*
- Upload current protocol sections
- Run grammar and consistency analysis
- Check ICH M11 compliance scores
- Get specific recommendations for improvements

## 🔧 Configuration

### Environment Variables
```env
# Required
OPENAI_API_KEY=sk-your-openai-key
QDRANT_URL=https://your-cluster.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-key

# Optional
GROQ_API_KEY=gsk_your-groq-key      # For free models
SERPER_API_KEY=your-serper-key      # For web search
```

### Cost Management
- **OpenAI Models**: Premium but highest quality
- **Groq Models**: Free tier available with good quality
- **Analysis Tools**: Completely free (no API calls)
- **Real-time Tracking**: Monitor usage and costs

## 📊 Features Deep Dive

### AI-Free Analysis Tools
Our analysis tools run completely locally without API calls, making them cost-effective for regular use:

- **Drug Information Extraction**: Uses pattern matching and medical NLP to identify drugs, dosages, and administration details
- **Grammar Checking**: Integrates with LanguageTool, filtered for medical terminology
- **ICH M11 Compliance**: Rule-based validation against international standards
- **Medical Terminology**: Consistency checking with standardized medical vocabulary

### Multi-Model AI Support
- **OpenAI GPT-4**: Highest quality, best for critical sections
- **OpenAI GPT-4-Turbo**: Faster processing, good quality
- **Groq Llama-3.3-70B**: Free tier available, excellent quality
- **Groq Llama-3.1-8B**: Fast and free, good for quick generations
- **Automatic Fallbacks**: Switches models on rate limits or errors

### Document Processing Pipeline
1. **Upload**: PDF/DOCX files accepted
2. **Extraction**: Text extracted with metadata preservation
3. **Chunking**: Intelligent text splitting (1024 tokens with overlap)
4. **Embedding**: OpenAI embeddings for semantic search
5. **Indexing**: Stored in Qdrant vector database
6. **Retrieval**: Context-aware search for AI generation

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork the Repository**
2. **Create a Feature Branch**: `git checkout -b feature/your-feature`
3. **Make Your Changes**: Follow our coding standards
4. **Add Tests**: Ensure your code is tested
5. **Update Documentation**: Keep docs current
6. **Submit a Pull Request**: Describe your changes clearly

### Development Guidelines
- **Code Style**: Follow PEP 8
- **Testing**: Add unit tests for new features
- **Documentation**: Update docstrings and user guides
- **Performance**: Consider API costs and response times

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Getting Help
- **Issues**: Report bugs or request features on GitHub
- **Documentation**: Check our comprehensive documentation
- **Community**: Join discussions in our GitHub Discussions

### Common Issues
- **API Rate Limits**: Use Groq models for free tier or implement delays
- **Document Processing**: Ensure PDF/DOCX files are text-searchable
- **Vector Database**: Check Qdrant connection and API key validity
- **Cost Management**: Monitor usage with built-in tracking tools

## 🔮 Roadmap

### Coming Soon
- **User Authentication**: Multi-user support with role-based access
- **Advanced Templates**: More ICH sections and custom templates
- **Batch Processing**: Generate multiple sections simultaneously
- **Enhanced Search**: Hybrid vector + keyword search
- **Mobile App**: Native mobile interface

### Long-term Vision
- **Regulatory Integration**: Direct submission to regulatory bodies
- **Collaborative Editing**: Real-time multi-user editing
- **Advanced Analytics**: Protocol quality metrics and benchmarking
- **API Marketplace**: Integration with external medical databases

---

**Ready to revolutionize your clinical trial protocol writing?** 

[Get Started Now](#quick-start) | [View Documentation](PROJECT.md) | [API Reference](http://localhost:8000/docs)
