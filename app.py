import os
from dotenv import load_dotenv
import streamlit as st
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# Load our secret environment keys
load_dotenv()

# ==============================================================================
# 1. UI/UX PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Smart Scholar AI", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #fcfcfc;}
    .stButton>button {background-color: #0078d4; color: white; border-radius: 8px; width: 100%;}
    .stChatInputContainer {border-radius: 8px;}
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# ==============================================================================
# 2. INITIALIZE CLIENT CONNECTIONS (Fully Patched for Streamlit Cloud SDK)
# ==============================================================================
@st.cache_resource
# ==============================================================================
# 2. INITIALIZE CLIENT CONNECTIONS (Streamlined for Cloud Container Compatibility)
# ==============================================================================
# Removed @st.cache_resource to prevent internal thread-hashing validation crashes
# ==============================================================================
# 2. INITIALIZE CLIENT CONNECTIONS (Production Cloud Fixed)
# ==============================================================================
def get_azure_clients():
    # Correctly resolve variables using simple fallbacks without hardcoding empty strings
    openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or st.secrets.get("AZURE_OPENAI_ENDPOINT")
    openai_key = os.getenv("AZURE_OPENAI_KEY") or st.secrets.get("AZURE_OPENAI_KEY")
    
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT") or st.secrets.get("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_KEY") or st.secrets.get("AZURE_SEARCH_KEY")
    search_index = os.getenv("AZURE_SEARCH_INDEX") or st.secrets.get("AZURE_SEARCH_INDEX")
    
    doc_endpoint = os.getenv("DOC_INTEL_ENDPOINT") or st.secrets.get("DOC_INTEL_ENDPOINT")
    doc_key = os.getenv("DOC_INTEL_KEY") or st.secrets.get("DOC_INTEL_KEY")

    # Clean, direct instantiation using resolved values
    openai_cl = AzureOpenAI(
        azure_endpoint=openai_endpoint,
        api_key=openai_key,
        api_version="2024-02-01"
    )
    
    search_cl = SearchClient(
        endpoint=search_endpoint,
        index_name=search_index,
        credential=AzureKeyCredential(search_key)
    )
    
    doc_cl = DocumentIntelligenceClient(
        endpoint=doc_endpoint, 
        credential=AzureKeyCredential(doc_key)
    )
    
    return openai_cl, search_cl, doc_cl

# Call initialization normally
openai_client, search_client, doc_client = get_azure_clients()

def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-3-small", 
        input=text
    )
    return response.data[0].embedding

# ==============================================================================
# 3. SIDEBAR / DOCUMENT UPLOADER LAYOUT
# ==============================================================================
with st.sidebar:
    st.title("⚙️ Management Panel")
    st.subheader("Upload Academic Content")
    
    uploaded_file = st.file_uploader("Drop your lecture notes or textbooks here", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Extract & Index Knowledge Base"):
            with st.spinner("Document Intelligence reading PDF structure..."):
                file_bytes = uploaded_file.getvalue()
                request_body = AnalyzeDocumentRequest(bytes_source=file_bytes)
                
                poller = doc_client.begin_analyze_document(
                    model_id="prebuilt-layout", 
                    body=request_body
                )
                result = poller.result()
                
                chunks = []
                current_chunk = ""
                for page in result.pages:
                    for line in page.lines:
                        current_chunk += line.content + " "
                        if len(current_chunk) > 1000:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
            with st.spinner("Converting text to vectors and saving to cloud index..."):
                documents = []
                for i, chunk in enumerate(chunks):
                    vector = get_embedding(chunk)
                    
                    # Perfectly matches your newly updated portal fields!
                    documents.append({
                        "id": f"uploaded_doc_{i}",
                        "chunk": chunk,     
                        "vector": vector    
                    })
                    
                search_client.upload_documents(documents=documents)
                st.success("Knowledge Base successfully updated!")

    st.markdown("---")
    st.caption("Built with Python, Azure AI Foundry, and Streamlit.")

# ==============================================================================
# 4. MAIN INTERACTIVE CHAT INTERFACE
# ==============================================================================
st.title("🎓 Smart Scholar: Intelligent Student Assistant")
st.write("Upload your syllabus or text notes in the sidebar, then ask questions below.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! I am ready to help you study. Drop your materials in the sidebar and ask me anything."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a question about your uploaded notes..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        with st.spinner("Searching through your notes..."):
            query_vector = get_embedding(user_query)
            
            from azure.search.documents.models import VectorizedQuery
            vector_query = VectorizedQuery(vector=query_vector, k_nearest_neighbors=3, fields="vector")
            
            results = search_client.search(
                search_text=user_query,
                vector_queries=[vector_query],
                top=3
            )
            
            context = "\n\n".join([r["chunk"] for r in results])
            
            system_prompt = (
                "You are an expert academic tutor. Answer the user's questions using only the context provided below. "
                "Be thorough, structured, and accurate. If the context doesn't contain the answer, tell the student honestly.\n\n"
                f"--- STUDY MATERIAL CONTEXT ---\n{context}"
            )
            
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *st.session_state.messages
                ],
                temperature=0.3
            )
            answer = response.choices[0].message.content
            
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})