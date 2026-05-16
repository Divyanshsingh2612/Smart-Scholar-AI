# Smart-Scholar-AI
Smart Scholar is an advanced Retrieval-Augmented Generation (RAG) web application designed to transform static academic materials into an interactive, learning experience. Built specifically for students, the application allows users to upload complex, lecture slides, or handwritten notes, and instantly query them using natural language.
# 🎓 Smart Scholar: AI-Powered Virtual Student Assistant (RAG System)

Smart Scholar is a web-based Retrieval-Augmented Generation (RAG) application engineered to transform static textbooks, multi-column research papers, and lecture notes into dynamic, conversational learning companions. Built using a secure, context-grounded framework, the assistant provides deterministic, source-verified answers directly from your syllabus, drastically mitigating standard AI hallucinations.

---

## 🛠️ Tech Stack & Azure Core Services

The platform integrates a production-grade Python stack seamlessly unified by the **Microsoft Foundry** ecosystem:

* **Front-end UI/UX (Streamlit):** Powers a responsive user interface featuring an intuitive sidebar for drag-and-drop document uploads, progress tracking spinners, and a persistent chat interface for multi-turn dialogue.
* **Document Parsing (Azure AI Document Intelligence):** Employs advanced layout parsing and deep-learning OCR models (`prebuilt-layout`) to read raw PDF streams, extracting structural text blocks and intricate tables while keeping the original formatting context intact.
* **Vector Database (Azure AI Search):** Configured with a highly optimized Hierarchical Navigable Small World ($HNSW$) vector profile running cosine similarity metrics. It indexes document fragments to allow fast conceptual matching rather than basic keyword searching.
* **Generative AI & Embeddings (Azure OpenAI Service):** Coordinates the conversational brain using `text-embedding-3-small` to convert textual chunks into discrete numerical data arrays ($1536$ dimensions) and `gpt-4o` to evaluate retrieved context and return precise academic summaries.

---

## 📂 End-to-End Pipeline

1.  **Ingestion:** A user drops academic course materials into the Streamlit sidebar.
2.  **Vectorization:** Azure Document Intelligence extracts raw data, chunking text into ~1000-character blocks, which Azure OpenAI then transforms into high-dimensional embeddings.
3.  **Indexing:** Vector mappings are written securely into the cloud search index under specialized schema definitions (`id`, `chunk`, `vector`).
4.  **Retrieval & Generation:** When a question is typed, the query is vectorized to fetch the three most conceptually relevant document snippets. These fragments act as tight bounding constraints within the system prompt, forcing the model to answer accurately based strictly on your text.

---

## 🚀 Local Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Smart-Scholar-AI.git](https://github.com/YOUR_GITHUB_USERNAME/Smart-Scholar-AI.git)
cd Smart-Scholar-AI
