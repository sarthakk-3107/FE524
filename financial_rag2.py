"""
Financial Filings RAG Analyst Assistant - Main Application
FE524 Project - Phase 1

Dependencies (using uv):
uv venv
.venv\\Scripts\\activate  # Windows
uv pip install -r requirements.txt

Or create a virtual environment:
uv venv
.venv\\Scripts\\activate  (Windows) or source .venv/bin/activate (Mac/Linux)
uv pip install -r requirements.txt

API key setup:
Your OpenAI API key must have access to:
- gpt-5-mini (for LLM)
- text-embedding-3-small (for embeddings)

Create a .env file in this directory with:
OPENAI_API_KEY=your-key-here

Then run: streamlit run financial_rag.py
"""

import os
import json
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from io import BytesIO

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
import numpy as np
from openai import OpenAI
import streamlit as st
from sec_edgar_downloader import Downloader

# Import document processor
try:
    from document_processor import SECDocumentProcessor
except ImportError:
    st.error("⚠️ document_processor.py not found! Please ensure it's in the same directory.")
    st.stop()

# Import visualization agent (lazy import - will import when needed)
try:
    from visualization_agent import FinancialVisualizationAgent
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    # Don't stop here - allow app to run without visualization

@dataclass
class DocumentChunk:
    """Represents a chunk of a financial document"""
    text: str
    source: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class FinancialRAGSystem:
    """Main RAG system for financial document analysis"""

    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        # Using OpenAI's text-embedding-3-small for better semantic search
        self.embedding_model_name = "text-embedding-3-small"
        self.chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
        self.collection = None
        self.bm25 = None
        self.chunks = []
        self.doc_processor = SECDocumentProcessor()

    def _truncate_text(self, text: str, max_chars: int = 8000) -> str:
        """Truncate text to fit within embedding model token limits.
        
        text-embedding-3-small has 8192 token limit.
        Using conservative estimate: ~1 token per 3 chars.
        8000 chars ≈ 2600 tokens (safe margin under 8192 limit).
        """
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI API (text-embedding-3-small)"""
        # Truncate texts to fit within model's token limit
        truncated_texts = [self._truncate_text(t) for t in texts]
        
        # Small batch size to avoid hitting API limits
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(truncated_texts), batch_size):
            batch = truncated_texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                # If batch fails, try one at a time with more aggressive truncation
                for text in batch:
                    try:
                        # Even more aggressive truncation for problematic texts
                        safe_text = text[:5000] if len(text) > 5000 else text
                        response = self.client.embeddings.create(
                            model=self.embedding_model_name,
                            input=[safe_text]
                        )
                        all_embeddings.append(response.data[0].embedding)
                    except Exception as inner_e:
                        # Last resort: very short text
                        response = self.client.embeddings.create(
                            model=self.embedding_model_name,
                            input=[text[:2000]]
                        )
                        all_embeddings.append(response.data[0].embedding)
        
        return all_embeddings

    def download_10k(self, ticker: str, email: str, num_filings: int = 1):
        """Download SEC 10-K filings for a company.
        
        Returns:
            tuple: (filepath, was_cached) - filepath to the filing, and whether it was already cached
        """
        filing_dir = f"sec-edgar-filings/{ticker}/10-K"
        
        # Check if filing already exists locally
        if os.path.exists(filing_dir):
            files = list(Path(filing_dir).rglob("*.txt"))
            if files:
                # Filing already downloaded, return existing file
                return str(files[0]), True  # True = already cached
        
        # Download if not found locally
        dl = Downloader("FE524-Project", email)
        dl.get("10-K", ticker, limit=num_filings)
        
        # Find the downloaded filing
        if os.path.exists(filing_dir):
            files = list(Path(filing_dir).rglob("*.txt"))
            if files:
                return str(files[0]), False  # False = newly downloaded
        return None, False

    def process_and_index_filing(self, filepath: str) -> int:
        """Process SEC filing and create searchable index"""
        # Process the filing
        sections, metadata = self.doc_processor.process_filing(filepath)

        # Create chunks from all sections
        all_chunks = []
        
        # Check parsing method and provide feedback
        parsing_method = metadata.get('parsing_method', 'unknown')
        
        if parsing_method == 'structured':
            st.success(f"✅ Successfully parsed {len(sections)} document sections")
        elif parsing_method == 'fallback_keyword':
            st.info(f"ℹ️ Used keyword-based parsing. Found {len(sections)} content sections.")
        elif parsing_method == 'full_document':
            st.warning("⚠️ Section headers not found. Using full document chunking.")

        if sections and len(sections) > 0:
            # Process each section
            for section in sections:
                section_chunks = self.doc_processor.chunk_section(section)

                # Convert to DocumentChunk objects
                for chunk_dict in section_chunks:
                    chunk = DocumentChunk(
                        text=chunk_dict['text'],
                        source=f"{chunk_dict['metadata']['section']} - {chunk_dict['metadata']['section_title']}",
                        metadata=chunk_dict['metadata']
                    )
                    all_chunks.append(chunk)
        else:
            # Final fallback: chunk the entire document
            raw_content = self.doc_processor.read_filing(filepath)
            clean_text = self.doc_processor.clean_html(raw_content)

            # Create chunks from full text (smaller chunks to fit embedding model limits)
            words = clean_text.split()
            chunk_size = 500
            overlap = 100

            for i in range(0, len(words), chunk_size - overlap):
                chunk_text = ' '.join(words[i:i + chunk_size])
                if len(chunk_text) > 100:
                    chunk = DocumentChunk(
                        text=chunk_text,
                        source=f"{metadata.get('company_name', 'Unknown')} 10-K",
                        metadata={
                            'company': metadata.get('company_name', 'Unknown'),
                            'filing_date': metadata.get('filing_date', ''),
                            'chunk_id': len(all_chunks)
                        }
                    )
                    all_chunks.append(chunk)

        if len(all_chunks) == 0:
            raise ValueError("No document chunks created. The file may be empty or corrupted.")

        # Index the chunks
        num_chunks = self.index_documents(all_chunks)
        return num_chunks, metadata

    def index_documents(self, chunks: List[DocumentChunk]):
        """Create vector and BM25 indexes for document chunks"""
        self.chunks = chunks

        # Create vector embeddings using OpenAI API
        texts = [chunk.text for chunk in chunks]
        embeddings = self.get_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = np.array(embedding)

        # Initialize ChromaDB collection
        try:
            self.chroma_client.delete_collection("financial_docs")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name="financial_docs",
            metadata={"hnsw:space": "cosine"}
        )

        # Add to ChromaDB
        self.collection.add(
            embeddings=embeddings,  # Already a list from OpenAI API
            documents=texts,
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )

        # Create BM25 index
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        return len(chunks)

    def hybrid_retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid retrieval using dense embeddings and BM25"""

        # Dense retrieval (semantic search) using OpenAI embeddings
        query_embedding = self.get_embeddings([query])[0]
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],  # Already a list from OpenAI API
            n_results=top_k * 2
        )

        # BM25 retrieval (keyword search)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top_indices = np.argsort(bm25_scores)[-top_k * 2:][::-1]

        # Combine and rerank
        chunk_scores = {}

        # Add dense results
        for idx, doc_id in enumerate(dense_results['ids'][0]):
            chunk_idx = int(doc_id.split('_')[1])
            chunk_scores[chunk_idx] = chunk_scores.get(chunk_idx, 0) + (1 - idx / (top_k * 2))

        # Add BM25 results
        for rank, idx in enumerate(bm25_top_indices):
            chunk_scores[idx] = chunk_scores.get(idx, 0) + (1 - rank / (top_k * 2))

        # Get top-k by combined score
        top_indices = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved_chunks = []
        for idx, score in top_indices:
            chunk = self.chunks[idx]
            retrieved_chunks.append({
                'text': chunk.text,
                'source': chunk.source,
                'metadata': chunk.metadata,
                'relevance_score': score
            })

        return retrieved_chunks

    def calculate_financial_metrics(self, context: str, query: str) -> Dict[str, Any]:
        """Use LLM to extract and calculate financial metrics"""

        calculation_prompt = f"""You are a financial analyst. Based on the following context, 
extract relevant financial data and perform calculations to answer the query.

Context:
{context}

Query: {query}

Provide:
1. Extracted financial numbers with their sources
2. Step-by-step calculations
3. Final computed metrics as a JSON object

Format your response as JSON:
{{
    "extracted_data": {{"item": value}},
    "calculations": ["step 1", "step 2"],
    "metrics": {{"metric_name": value}},
    "insights": ["insight 1", "insight 2"]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",  # Using GPT-5-mini for advanced financial analysis
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at extracting data and performing calculations."},
                    {"role": "user", "content": calculation_prompt}
                ]
            )

            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "error": str(e),
                "calculations": [],
                "metrics": {},
                "insights": []
            }

    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """Generate final answer with citations"""

        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source {i+1}: {chunk['source']}]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        # Calculate metrics if needed
        metrics_result = self.calculate_financial_metrics(context, query)

        # Generate narrative answer
        answer_prompt = f"""You are a financial analyst assistant. Answer the user's question based on the provided context.

Context from SEC 10-K filings:
{context}

Calculated Metrics:
{json.dumps(metrics_result.get('metrics', {}), indent=2)}

Question: {query}

Provide a clear, professional answer that:
1. Directly answers the question
2. Cites specific sources (e.g., "According to [Source 1]...")
3. Includes relevant metrics and calculations
4. Highlights key insights

Keep your answer concise but comprehensive."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-5-mini",  # Using GPT-5-mini for advanced financial analysis
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst providing evidence-based insights."},
                    {"role": "user", "content": answer_prompt}
                ]
            )

            return {
                'answer': response.choices[0].message.content,
                'retrieved_chunks': retrieved_chunks,
                'metrics': metrics_result.get('metrics', {}),
                'calculations': metrics_result.get('calculations', []),
                'insights': metrics_result.get('insights', [])
            }
        except Exception as e:
            return {
                'error': f"Error generating answer: {str(e)}",
                'retrieved_chunks': retrieved_chunks,
                'metrics': {},
                'calculations': [],
                'insights': []
            }

    def query(self, question: str) -> Dict[str, Any]:
        """Main query interface"""
        if not self.chunks:
            return {"error": "No documents indexed. Please load documents first."}

        # Retrieve relevant chunks
        retrieved = self.hybrid_retrieve(question, top_k=5)

        # Generate answer
        result = self.generate_answer(question, retrieved)

        return result


def generate_pdf(result: Dict[str, Any], query: str, metadata: Dict = None) -> BytesIO:
    """Generate PDF from analysis result"""
    if not PDF_AVAILABLE:
        return None
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Financial Analysis Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Metadata
    if metadata:
        meta_text = f"<b>Company:</b> {metadata.get('company_name', 'N/A')}<br/>"
        meta_text += f"<b>Ticker:</b> {metadata.get('ticker', 'N/A')}<br/>"
        meta_text += f"<b>Filing Date:</b> {metadata.get('filing_date', 'N/A')}<br/>"
        meta_text += f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(meta_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Query
    query_style = ParagraphStyle(
        'QueryStyle',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12
    )
    story.append(Paragraph(f"<b>Question:</b> {query}", query_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Answer
    if 'answer' in result:
        story.append(Paragraph("<b>Answer:</b>", styles['Heading2']))
        # Clean HTML and format text
        answer_text = result['answer'].replace('\n', '<br/>')
        story.append(Paragraph(answer_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Metrics
    if result.get('metrics'):
        story.append(Paragraph("<b>Computed Metrics:</b>", styles['Heading2']))
        metrics_data = [['Metric', 'Value']]
        for metric_name, metric_value in result['metrics'].items():
            if isinstance(metric_value, dict):
                for key, val in metric_value.items():
                    if val is not None:
                        label = key.replace('_', ' ').title()
                        metrics_data.append([label, str(val)])
            elif metric_value is not None:
                metrics_data.append([metric_name, str(metric_value)])
        
        if len(metrics_data) > 1:
            metrics_table = Table(metrics_data, colWidths=[4*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 0.3*inch))
    
    # Calculations
    if result.get('calculations'):
        story.append(Paragraph("<b>Calculations:</b>", styles['Heading2']))
        for i, calc in enumerate(result['calculations'], 1):
            story.append(Paragraph(f"{i}. {calc}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Insights
    if result.get('insights'):
        story.append(Paragraph("<b>Key Insights:</b>", styles['Heading2']))
        for insight in result['insights']:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Sources
    if result.get('retrieved_chunks'):
        story.append(PageBreak())
        story.append(Paragraph("<b>Sources:</b>", styles['Heading2']))
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            source_text = f"<b>Source {i}:</b> {chunk.get('source', 'N/A')}<br/>"
            source_text += f"<i>Relevance: {chunk.get('relevance_score', 0):.2%}</i><br/><br/>"
            source_text += chunk.get('text', '')[:500] + "..."
            story.append(Paragraph(source_text, styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Financial Analyst", 
        page_icon="📊", 
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None  # Remove menu items (burger menu)
    )
    
    # Custom CSS to hide deploy button and style the app
    st.markdown("""
    <style>
        /* Hide default Streamlit header completely */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        
        /* Hide deploy button */
        .stDeployButton,
        [data-testid="stAppDeployButton"] {
            display: none !important;
        }
        
        /* Hide hamburger menu */
        #MainMenu,
        [data-testid="stMainMenu"] {
            display: none !important;
        }
        
        footer {
            visibility: hidden;
        }
        
        /* Remove top padding since we're hiding the header */
        .stApp > div:first-child {
            padding-top: 0 !important;
        }
        
        /* Adjust main content area */
        .main .block-container {
            padding-top: 0.5rem;
        }
        
        /* Modern color scheme */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Style headers */
        h1 {
            color: #2c3e50;
            font-weight: 600;
        }
        
        h2, h3 {
            color: #34495e;
            font-weight: 500;
        }
        
        /* Style buttons */
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 6px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
            transition: background-color 0.3s;
        }
        
        .stButton>button:hover {
            background-color: #2980b9;
        }
        
        /* Style text inputs */
        .stTextInput>div>div>input {
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        
        /* Style text area */
        .stTextArea>div>div>textarea {
            border-radius: 6px;
            border: 1px solid #ddd;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #ffffff;
        }
        
        /* Metric cards */
        [data-testid="stMetricValue"] {
            color: #2c3e50;
            font-weight: 600;
        }
        
        /* Sample questions buttons */
        .sample-question-btn {
            background-color: #ecf0f1;
            color: #2c3e50;
            border: 1px solid #bdc3c7;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            margin: 0.25rem;
            font-size: 0.9rem;
            transition: all 0.3s;
        }
        
        .sample-question-btn:hover {
            background-color: #d5dbdb;
            border-color: #95a5a6;
        }
        
        /* Custom header styling */
        .app-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem 2rem;
            margin: -1rem -1rem 2rem -1rem;
            border-radius: 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .app-header h1 {
            color: white;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
            text-align: center;
        }
        
        .app-header p {
            color: rgba(255,255,255,0.9);
            margin: 0.5rem 0 0 0;
            text-align: center;
            font-size: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Custom header
    st.markdown("""
    <div class="app-header">
        <h1>Financial Analyst</h1>
        <p>AI-Powered Financial Document Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize visualization availability in session state
    if 'viz_available' not in st.session_state:
        st.session_state['viz_available'] = VISUALIZATION_AVAILABLE

    # Check for API key in environment
    api_key = os.environ.get('OPENAI_API_KEY')

    if not api_key:
        st.error("❌ OpenAI API key not found!")
        st.info("""
        Create a `.env` file in the project folder with:
        ```
        OPENAI_API_KEY=your-key-here
        ```
        Then restart the app.
        """)
        st.stop()

    # Sidebar for configuration
    with st.sidebar:
        st.header("📁 Load Company Filing")
        
        # Popular ticker suggestions
        popular_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "V", "JNJ"]
        
        # Ticker input with autocomplete suggestions
        ticker_input = st.text_input("Company Ticker", "AAPL", help="Enter company ticker symbol", key="ticker_input")
        
        # Show suggestions if user is typing
        if ticker_input and len(ticker_input) > 0:
            matching_tickers = [t for t in popular_tickers if t.startswith(ticker_input.upper())]
            if matching_tickers:
                st.caption(f"💡 Suggestions: {', '.join(matching_tickers[:5])}")
        
        ticker = ticker_input.upper() if ticker_input else "AAPL"
        email = st.text_input("Your Email", "user@email.com", help="Required by SEC for downloading")

        if st.button("📥 Load 10-K Filing", type="primary", use_container_width=True):
            with st.spinner(f"Loading {ticker} 10-K..."):
                try:
                    # Initialize or get RAG system
                    if 'rag_system' not in st.session_state:
                        st.session_state['rag_system'] = FinancialRAGSystem(api_key)
                    
                    # Initialize visualization agent if not already done
                    if 'viz_agent' not in st.session_state and st.session_state['viz_available']:
                        try:
                            st.session_state['viz_agent'] = FinancialVisualizationAgent(
                                st.session_state['rag_system'],
                                OpenAI(api_key=api_key)
                            )
                        except Exception as e:
                            st.warning(f"Could not initialize visualization agent: {str(e)}")
                            st.session_state['viz_available'] = False

                    rag_system = st.session_state['rag_system']

                    # Check for existing or download filing
                    filepath, was_cached = rag_system.download_10k(ticker, email)

                    if not filepath:
                        st.error("Could not find filing. Check the ticker symbol.")
                    else:
                        # Show whether using cached or newly downloaded
                        if was_cached:
                            st.info(f"📂 Using cached {ticker} filing")
                        else:
                            st.success(f"📥 Downloaded {ticker} filing")
                        
                        # Process and index
                        num_chunks, metadata = rag_system.process_and_index_filing(filepath)

                        # Store metadata in session
                        st.session_state['metadata'] = metadata
                        st.session_state['ticker'] = ticker

                        st.success(f"✅ Ready! {num_chunks} sections indexed")
                        st.balloons()

                except Exception as e:
                    st.error(f"Error: {str(e)}")
        

        st.markdown("---")
        st.subheader("📄 Loaded Document")
        if 'metadata' in st.session_state:
            metadata = st.session_state['metadata']
            st.write(f"**{metadata.get('company_name', st.session_state.get('ticker', 'Unknown'))}**")
            st.caption(f"Filed: {metadata.get('filing_date', 'N/A')}")
        else:
            st.caption("No document loaded")

    # Show welcome message if no document loaded
    if 'rag_system' not in st.session_state:
        st.info("👈 **Get started:** Enter a company ticker in the sidebar and click 'Load 10-K Filing'")
        st.markdown("---")
    
    # Main query interface
    st.header("💬 Ask a Question")

    # Get query from session state or text area
    default_query = st.session_state.get('query', '')

    query = st.text_area(
        "What would you like to know?",
        value=default_query,
        height=80,
        placeholder="e.g., What was the total revenue? What are the main risks?",
        key="query_input"
    )
    
    # Sample questions displayed horizontally below the text area
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**💡 Sample Questions:**", unsafe_allow_html=True)
    sample_questions = [
        "What was the total revenue?",
        "Calculate the operating margin",
        "What are the main risk factors?",
        "Compare R&D expenses to revenue",
        "What are the key business segments?"
    ]
    
    # Display sample questions in horizontal layout
    cols = st.columns(len(sample_questions))
    for idx, q in enumerate(sample_questions):
        with cols[idx]:
            if st.button(q, key=f"sample_{q}", use_container_width=True):
                st.session_state['query'] = q
                st.session_state['auto_analyze'] = True
                st.rerun()

    if st.session_state.get('viz_available', False):
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            generate_graph_btn = st.button("📊 Generate Graph", use_container_width=True)
        with col3:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
    else:
        col1, col2 = st.columns([1, 5])
        with col1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.clear()
                st.rerun()
        generate_graph_btn = False

    # Check if auto-analyze was triggered by sample question
    auto_analyze = st.session_state.pop('auto_analyze', False)
    
    # Process query (either from button click or auto-analyze)
    if (analyze_btn or auto_analyze) and query:
        if 'rag_system' not in st.session_state:
            st.warning("Please load a 10-K filing first using the sidebar.")
        else:
            rag_system = st.session_state['rag_system']

            with st.spinner("Analyzing..."):
                result = rag_system.query(query)

                if 'error' in result:
                    st.error(f"❌ {result['error']}")
                else:
                    # Store result for PDF generation
                    st.session_state['last_result'] = result
                    st.session_state['last_query'] = query
                    
                    # Display answer with PDF download button
                    col_title, col_pdf = st.columns([4, 1])
                    with col_title:
                        st.header("📝 Answer")
                    with col_pdf:
                        if PDF_AVAILABLE:
                            metadata = st.session_state.get('metadata', {})
                            if metadata and 'ticker' not in metadata:
                                metadata['ticker'] = st.session_state.get('ticker', 'N/A')
                            pdf_buffer = generate_pdf(result, query, metadata)
                            if pdf_buffer:
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_buffer,
                                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    use_container_width=True
                                )
                        else:
                            st.caption("PDF export unavailable")
                    
                    st.markdown(result['answer'])

                    # Display metrics
                    if result.get('metrics') and len(result['metrics']) > 0:
                        st.header("📊 Computed Metrics")
                        
                        # Flatten nested dictionaries and handle different value types
                        flat_metrics = []
                        for metric_name, metric_value in result['metrics'].items():
                            if isinstance(metric_value, dict):
                                # If value is a dict, create separate metrics for each key-value pair
                                for key, val in metric_value.items():
                                    if val is not None:  # Skip None values
                                        # Format the key as a readable label
                                        label = key.replace('_', ' ').replace('millions USD', 'M USD').title()
                                        flat_metrics.append((label, val))
                            elif metric_value is not None:
                                # Regular metric value
                                flat_metrics.append((metric_name, metric_value))
                        
                        if flat_metrics:
                            metric_cols = st.columns(min(len(flat_metrics), 4))
                            for idx, (metric_label, metric_value) in enumerate(flat_metrics):
                                with metric_cols[idx % 4]:
                                    # Format the value appropriately
                                    if isinstance(metric_value, (int, float)):
                                        if abs(metric_value) >= 1000000:
                                            formatted_value = f"${metric_value/1000000:.2f}M"
                                        elif abs(metric_value) >= 1000:
                                            formatted_value = f"${metric_value/1000:.2f}K"
                                        else:
                                            formatted_value = f"${metric_value:,.0f}"
                                        st.metric(label=metric_label, value=formatted_value)
                                    else:
                                        st.metric(label=metric_label, value=str(metric_value))

                    # Display calculations
                    if result.get('calculations') and len(result['calculations']) > 0:
                        st.header("🧮 Calculations")
                        for i, calc in enumerate(result['calculations'], 1):
                            st.markdown(f"{i}. {calc}")

                    # Display insights
                    if result.get('insights') and len(result['insights']) > 0:
                        st.header("💡 Key Insights")
                        for insight in result['insights']:
                            st.markdown(f"- {insight}")
                    
                    # Offer to generate graph if metrics are available
                    if result.get('metrics') and len(result['metrics']) > 0 and st.session_state.get('viz_available', False) and 'viz_agent' in st.session_state:
                        st.markdown("---")
                        if st.button("📊 Generate Graph for This Query", key="auto_graph_btn"):
                            viz_agent = st.session_state['viz_agent']
                            with st.spinner("Generating visualization..."):
                                try:
                                    viz_result = viz_agent.run(query)
                                    
                                    st.header("📊 Visualization")
                                    st.plotly_chart(viz_result.chart, use_container_width=True)
                                    
                                    if viz_result.narrative:
                                        st.markdown("### 📝 Chart Description")
                                        st.markdown(viz_result.narrative)
                                    
                                    if viz_result.insights and len(viz_result.insights) > 0:
                                        st.markdown("### 💡 Key Insights")
                                        for insight in viz_result.insights:
                                            st.markdown(f"- {insight}")
                                    
                                    if not viz_result.data.empty:
                                        with st.expander("📋 View Data Table"):
                                            st.dataframe(viz_result.data, use_container_width=True)
                                except Exception as e:
                                    st.error(f"❌ Error generating visualization: {str(e)}")

                    # Display retrieved chunks
                    st.header("📄 Sources")

                    for i, chunk in enumerate(result['retrieved_chunks'], 1):
                        with st.expander(
                            f"📑 Source {i}: {chunk['source']} | Relevance: {chunk['relevance_score']:.2%}",
                            expanded=(i == 1)
                        ):
                            st.markdown(chunk['text'])
                            st.caption(f"Company: {chunk['metadata'].get('company', 'N/A')} | Section: {chunk['metadata'].get('section', 'N/A')}")
    
    # Process graph generation
    if generate_graph_btn and query and st.session_state.get('viz_available', False):
        if 'rag_system' not in st.session_state:
            st.warning("Please load a 10-K filing first using the sidebar.")
        elif 'viz_agent' not in st.session_state:
            st.warning("Visualization agent not initialized. Please reload the 10-K filing.")
        else:
            viz_agent = st.session_state['viz_agent']
            
            with st.spinner("Generating visualization..."):
                try:
                    viz_result = viz_agent.run(query)
                    
                    # Display the chart
                    st.header("📊 Visualization")
                    st.plotly_chart(viz_result.chart, use_container_width=True)
                    
                    # Display narrative
                    if viz_result.narrative:
                        st.markdown("### 📝 Chart Description")
                        st.markdown(viz_result.narrative)
                    
                    # Display insights
                    if viz_result.insights and len(viz_result.insights) > 0:
                        st.markdown("### 💡 Key Insights")
                        for insight in viz_result.insights:
                            st.markdown(f"- {insight}")
                    
                    # Display data table
                    if not viz_result.data.empty:
                        with st.expander("📋 View Data Table"):
                            st.dataframe(viz_result.data, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating visualization: {str(e)}")
                    st.info("💡 Try refining your query to be more specific about the financial data you want to visualize (e.g., 'Show revenue trends from 2021 to 2023').")

    # Footer
    st.markdown("---")
    st.caption("Financial RAG Analyst • Powered by OpenAI • Data from SEC EDGAR")

if __name__ == "__main__":
    main()
