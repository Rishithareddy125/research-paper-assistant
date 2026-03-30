#  PaperMind — AI Research Paper Assistant

PaperMind is an AI-powered web application that transforms how students and 
researchers interact with academic papers. Instead of reading through lengthy 
PDFs manually, users can simply upload any research paper and have a natural 
conversation with it — getting instant, accurate answers to any question.

---

##  Live Demo
Coming soon...

---

##  Features
-  Upload and chat with multiple research papers simultaneously
-  Chat interface with full conversation memory
-  Smart chunk retrieval using FAISS vector search
-  MMR search for diverse, high-quality answers
-  Source page references shown with every answer
-  Save and reload FAISS indexes to avoid reprocessing
-  Powered by Groq (free and fast LLM API)
-  Beautiful dark themed chat interface

---

##  Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (LLaMA 3.3 70B) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| PDF Parsing | PyMuPDF |
| Orchestration | LangChain |

---

##  Project Structure
```
research-paper-assistant/
├── app.py              # Main Streamlit application
├── rag_pipeline.py     # RAG pipeline (chunking, embeddings, QA chain)
├── utils.py            # Helper utilities
├── requirements.txt    # Python dependencies
└── .env                # API keys (not uploaded to GitHub)
```

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/Rishithareddy125/research-paper-assistant.git
cd research-paper-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free API key at  https://console.groq.com

**4. Run the app**
```bash
streamlit run app.py
```

**5. Open in browser**
```
http://localhost:8501
```

---

##  How to Get a Free Groq API Key
1. Go to https://console.groq.com
2. Sign in with Google or GitHub
3. Click **API Keys** → **Create API Key**
4. Copy the key and paste it in your `.env` file

---

##  How It Works

1. **Upload** a research paper PDF
2. The app **chunks** the PDF into small pieces
3. Each chunk is **embedded** into vectors using HuggingFace
4. Vectors are stored in a **FAISS** index
5. When you ask a question, the most relevant chunks are **retrieved**
6. Retrieved chunks are sent to **Groq LLaMA 3.3 70B** to generate an answer
7. The answer is displayed with **source page references**

---

##  Use Cases
-  Students understanding complex research papers quickly
-  Researchers comparing methodologies across multiple papers
-  Anyone extracting specific information from academic documents
-  Professionals staying up to date with industry research

---

##  Author
**Rishitha Reddy**  
GitHub: [@Rishithareddy125](https://github.com/Rishithareddy125)

