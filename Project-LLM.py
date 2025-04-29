#region import libraries
import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel

from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

#endregion


#region models and helper functions with initialization
#region lora

# @st.cache_resource
# def load_model():
#     base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
#     base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
#     tokenizer = AutoTokenizer.from_pretrained("Frontend\\tinyllama_lora_muslim_family_law")

#     model = PeftModel.from_pretrained(base_model, "Frontend\\tinyllama_lora_muslim_family_law")
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

#endregion

#region RAG
# @st.cache_resource
# def load_rag_model():
#     # Load the Hugging Face model and tokenizer
#     # model_name = "deepseek-ai/deepseek-llm-7b-chat"
#     # tokenizer = AutoTokenizer.from_pretrained(model_name)
#     # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
#     model_name = "deepseek-ai/deepseek-llm-7b-chat"

#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#     )

#     model = AutoModelForCausalLM.from_pretrained(
#         model_name,
#         quantization_config=bnb_config,
#         device_map="auto"
#     )

#     tokenizer = AutoTokenizer.from_pretrained(model_name)
    
#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=1024,
#         temperature=0.7,
#         top_p=0.95,
#         repetition_penalty=1.15
#     )
    
#     # Wrap the pipeline with HuggingFacePipeline for LangChain integration
#     llm = HuggingFacePipeline(pipeline=pipe)

#     # Load the Chroma vector store
#     file_path = "raw_data\\image_pdf\\family_law_manual.pdf"
#     persistent_directory = os.path.join("db", "chroma_db")
    
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
#     loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
#     documents = loader.load()

#     # Split documents into chunks and create embeddings
#     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     document_chunks = text_splitter.split_documents(documents)

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#     vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory=persistent_directory)

#     return llm, vector_store

# model, tokenizer = load_rag_model()

@st.cache_resource
def load_rag_model():
    # Setup Gemini
    genai.configure(api_key="AIzaSyBmpIwMg_LnNnynv6R0YW7430BJTdUX1iI")

    # Load documents
    file_path = "raw_data\\image_pdf\\family_law_manual.pdf"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

    loader = PyPDFLoader(file_path)
    print("Loading PDF Loader")
    documents = loader.load()
    print("Documents loaded successfully.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    print("Splitting documents...")
    document_chunks = text_splitter.split_documents(documents)
    print("Documents split successfully.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embeddings loaded successfully.")
    vector_store = Chroma.from_documents(document_chunks, embeddings, persist_directory=None)
    print("Vector store loaded successfully.")
    return vector_store  # No model return now


#endregion
# Helper function to generate model response

# def generate_response(prompt_text):
#     inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
#     outputs = model.generate(**inputs, max_new_tokens=200)
#     reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return reply

# def generate_rag_response(prompt_text, vector_store, llm):
#     # Retrieve relevant documents from the vector store
#     retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
#     relevant_docs = retriever.invoke(prompt_text)

#     # Combine the query and the relevant document contents for input to the model
#     combined_input = (
#         f"Here are some documents that might help answer the question: {prompt_text}\n\n"
#         + "\n\n".join([doc.page_content for doc in relevant_docs])
#         + "\n\nPlease base your response on the documents provided and keep it as simple as possible. If the answer cannot be found within them, use your own knowledge."
#     )

#     # Format the prompt for the model
#     formatted_prompt = tokenizer.apply_chat_template(
#         [{"role": "system", "content": "You are a helpful legal assistant."},
#          {"role": "user", "content": combined_input}],
#         tokenize=False, add_generation_prompt=True
#     )

#     # Generate and return the response from the model
#     result = llm.invoke(formatted_prompt)
#     return result

def generate_rag_response(prompt_text, vector_store):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    relevant_docs = retriever.invoke(prompt_text)

    combined_input = (
        f"You are a legal assistant. Based on the following documents, answer the question:\n\n"
        + "\n\n".join([doc.page_content for doc in relevant_docs])
        + f"\n\nQuestion: {prompt_text}\nAnswer in simple, easy language."
    )

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(combined_input)

    return response.text

#endregion

#region streamlit uilibraries
# ─────────────────────────────────────────────────────────────────────────────
# 1) Include Font Awesome (for avatar/icon) – optional
st.markdown(
    '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css" rel="stylesheet">',
    unsafe_allow_html=True,
)
#endregion

# ─────────────────────────────────────────────────────────────────────────────
# 2) Session-state initialization
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {1: []}     # chat_id → list of (role, text)
if "active_chat" not in st.session_state:
    st.session_state.active_chat = 1

# ─────────────────────────────────────────────────────────────────────────────
# 3) Sidebar: select or create chats
with st.sidebar:
    st.title("WakeelGPT")
    st.header("💬 Chats")

    # List existing chats
    chat_ids = list(st.session_state.chat_sessions.keys())
    choice = st.radio(
        "Select chat",
        chat_ids,
        index=chat_ids.index(st.session_state.active_chat),
    )
    st.session_state.active_chat = choice

    st.markdown("---")
    if st.button("➕ New Chat"):
        new_id = max(chat_ids) + 1
        st.session_state.chat_sessions[new_id] = []
        st.session_state.active_chat = new_id
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["💼 Legal Consulting", "📄 Draft Generator","Citations"])
with tab1:
    # # Container for messages
    
    # chat_container = st.container()

    # # # Display history
    # # for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
    # #     with chat_container.chat_message(role):
    # #         chat_container.write(text)





    # # # with st.sidebar:
    # # messages = st.container()
    # # if prompt := st.chat_input(placeholder="Your message",accept_file=True,key="consulting"):
    # #     messages.chat_message("user").write(prompt.text)
    # #     messages.chat_message("assistant",avatar=":material/gavel:").write(f"Echo: {prompt.text}")
    # for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
    #     with chat_container.chat_message(role):
    #         chat_container.write(text)

    #     # Chat input
    #     prompt = st.chat_input(placeholder="Your message", key=f"chat_input_{key_suffix}")
    #     if prompt:
    #         # Save user message
    #         st.session_state.chat_sessions[st.session_state.active_chat].append(("user", prompt))

    #         with chat_container.chat_message("user"):
    #             st.write(prompt)

    #         # Generate response
    #         reply = generate_response(prompt)

    #         # Save assistant message
    #         st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

    #         with chat_container.chat_message("assistant", avatar=":material/gavel:"):
    #             st.write(reply)

    chat_container = st.container()

    # Display history
    for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
        with chat_container:
            st.chat_message(role).write(text)

    # Chat input outside the container for visibility
    prompt = st.chat_input(placeholder="Your message", key=f"chat_input_{st.session_state.active_chat}")
    if prompt:
        # Save user message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("user", prompt))

        with chat_container:
            st.chat_message("user").write(prompt)

        # Generate response
        # reply = generate_response(prompt)

        # Save assistant message
        # st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

        with chat_container:
            pass
            # st.chat_message("assistant", avatar=":material/gavel:").write(reply)

with tab2:
    # # Container for messages
    # chat_container = st.container()

    # # Display history
    # for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
    #     with chat_container.chat_message(role):
    #         chat_container.write(text)





    # # with st.sidebar:
    # messages = st.container()
    # if prompt := st.chat_input(placeholder="Your message",accept_file=True,key="draft"):
    #     messages.chat_message("user").write(prompt.text)
    #     messages.chat_message("assistant",avatar=":material/gavel:").write(f"Echo: {prompt.text}")

    chat_container = st.container()

    # Initialize the chat session for Tab 2
    if st.session_state.active_chat not in st.session_state.chat_sessions:
        st.session_state.chat_sessions[st.session_state.active_chat] = []

    # Display chat history
    for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
        with chat_container:
            st.chat_message(role).write(text)

    # User input field for the query
    prompt = st.chat_input(placeholder="Ask about family law in Pakistan", key=f"chat_input_t2_{st.session_state.active_chat}")
    
    if prompt:
        # Save user message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("user", prompt))

        with chat_container:
            st.chat_message("user").write(prompt)

        # Load the model and vector store only for Tab 2
        vector_store = load_rag_model()

        # Generate response using the RAG function
        reply = generate_rag_response(prompt, vector_store)

        # Save assistant message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

        with chat_container:
            st.chat_message("assistant", avatar=":material/gavel:").write(reply)

with tab3:
    # Container for messages
    chat_container = st.container()

    # Display history
    for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
        with chat_container.chat_message(role):
            chat_container.write(text)





    # with st.sidebar:
    messages = st.container()
    if prompt := st.chat_input(placeholder="Your message",accept_file=True,key="citations"):
        messages.chat_message("user").write(prompt.text)
        messages.chat_message("assistant",avatar=":material/gavel:").write(f"Echo: {prompt.text}")