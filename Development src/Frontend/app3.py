import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Page config
st.set_page_config(page_title="Wakeel - Legal Assistant", layout="wide")

# Sidebar
with st.sidebar:
    st.title("⚖️ Wakeel")
    if st.button("➕ New Chat"):
        st.session_state["chat_history"] = []
    st.text_input("🔍 Search chats")
    st.write("---")
    st.subheader("Manage Chats")
    st.button("📁 View All")
    st.button("🗑️ Clear All")
    st.write("---")
    st.subheader("User Profile & Settings")
    st.selectbox("Preferred Language", ["English", "Urdu", "Both"])
    st.button("⚙️ Settings")
    st.write("Logged in as: **Your Name**")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Load the LoRA model and tokenizer
@st.cache_resource
def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, "tinyllama_lora_muslim_family_law")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return model.eval(), tokenizer



model, tokenizer = load_model()

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
import streamlit as st
from datetime import datetime


file_path = "raw_data/image_pdf/family_law_manual.pdf"  # Adjust path if necessary
persistent_directory = os.path.join("db", "chroma_db")

@st.cache_resource
def load_rag_model():
    print("Initializing RAG model...")
    
    # Load document (PDF or text)
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)
    
    # Split document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = loader.load()
    document_chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Create and persist vector store
    vector_store = Chroma.from_documents(
        document_chunks, embeddings, persist_directory=persistent_directory)
    vector_store.persist()

    return vector_store, embeddings

# Load and initialize the model and vector store
vector_store, embeddings = load_rag_model()

# Tabs for Consulting and Legal Drafts
tab1, tab2, tab3 = st.tabs(["💼 Legal Consulting", "📄 Draft Generator","Citations"])

# Tab 1: Legal Consulting
with tab1:
    st.title("💬 Wakeel - Your AI Legal Consultant")
    chat_placeholder = st.container()

    with chat_placeholder:
        for chat in st.session_state["chat_history"]:
            if chat["role"] == "user":
                st.markdown(f"**🧑‍💼 You:** {chat['content']}")
            else:
                st.markdown(f"**🤖 Wakeel:** {chat['content']}")
                st.write(f"_{chat['time']}_")
                st.button("👍", key=f"like_{chat['time']}")
                st.button("👎", key=f"dislike_{chat['time']}")
                st.download_button(
                                    "📥 Download PDF",
                                    chat["content"],
                                    file_name="legal_draft.pdf",
                                    key=f"download_{chat['time']}")  # Unique key for each button


    st.write("---")

    st.subheader("Ask Your Question")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        uploaded_file = st.file_uploader("📎", label_visibility="collapsed")
    with col2:
        user_input = st.text_area("Type your query", label_visibility="collapsed", height=70)
    with col3:
        voice_input = st.button("🎤")
    send = st.button("Send", key="send_consulting")

    if send and user_input:
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Generate response using LoRA model
        inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print("Raw AI Response:", ai_response)
        
        st.session_state.chat_history.append({
            "role": "ai",
            "content": ai_response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.rerun()

# Tab 2: Draft Generator
# with tab2:
#     st.title("📄 Draft a Legal Document")
#     draft_input = st.text_area("Describe the document you want to generate", height=150)
#     generate = st.button("Generate Draft")

#     if generate and draft_input:
#         # Mock Draft Output
#         draft_output = f"**Draft for:** {draft_input}\n\nThis is your AI-generated legal document."
#         st.markdown(draft_output)
#         st.download_button("📥 Download Draft", draft_output, file_name="legal_draft.txt")
#         st.session_state.chat_history.append({
#             "role": "ai",
#             "content": draft_output,
#             "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         })


# Tab 2: Draft Generation
with tab2:
    st.title("📑 Legal Draft Generation")
    
    # Create a container for chat history
    draft_chat_placeholder = st.container()
    
    with draft_chat_placeholder:
        for chat in st.session_state["chat_history"]:
            if chat["role"] == "user":
                st.markdown(f"**🧑‍💼 You:** {chat['content']}")
            else:
                st.markdown(f"**🤖 Wakeel:** {chat['content']}")
                st.write(f"_{chat['time']}_")
                st.button("👍", key=f"like_{chat['time']}")
                st.button("👎", key=f"dislike_{chat['time']}")
                st.download_button(
                    "📥 Download PDF",
                    chat["content"],
                    file_name="legal_draft.pdf",
                    key=f"download_{chat['time']}"
                )

    st.write("---")

    st.subheader("Generate Your Legal Draft")
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        uploaded_file = st.file_uploader("📎", label_visibility="collapsed")
    with col2:
        user_input = st.text_area("Describe your draft requirements", label_visibility="collapsed", height=70)
    with col3:
        voice_input = st.button("🎤")
    generate_button = st.button("Generate Draft", key="generate_draft")

    if generate_button and user_input:
        # Save the user's query to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Query the vector store for relevant documents based on the input query
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
        relevant_docs = retriever.invoke(user_input)

        # Combine the user's query and relevant documents for generating the draft
        combined_input = (
            "Here are some documents that might help with drafting: "
            + user_input
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\nPlease generate the legal draft based on the provided information."
        )

        # Pass the combined input to the LLM for generation
        inputs = tokenizer(combined_input, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, top_p=0.9, temperature=0.7)
        ai_response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        # Save the AI response to chat history as a draft
        st.session_state.chat_history.append({
            "role": "ai",
            "content": ai_response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Display the generated draft
        st.markdown("### Generated Legal Draft:")
        st.write(ai_response)
        
        # Optionally allow users to download the generated draft as a PDF
        st.download_button(
            "📥 Download Draft",
            ai_response,
            file_name="legal_draft.pdf",
            key="download_draft"
        )
        
        st.rerun()

with tab3:
    pass