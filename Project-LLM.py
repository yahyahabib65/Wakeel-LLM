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
from fpdf import FPDF
import datetime
from docx import Document


#endregion


#region models and helper functions with initialization
#region lora

@st.cache_resource
def load_model():
    base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Frontend\\tinyllama_lora_muslim_family_law")

    model = PeftModel.from_pretrained(base_model, "Frontend\\tinyllama_lora_muslim_family_law")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

#endregion

#region RAG
# Load the RAG model and vector store
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

def generate_response(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply


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
        reply = generate_response(prompt)
        reply = "This is a placeholder response. Please implement the actual model response generation."

        # Save assistant message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

        with chat_container:
            # pass
            st.chat_message("assistant", avatar=":material/gavel:").write(reply)

with tab3:

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
        reply = "This is a placeholder response. Please implement the actual RAG response generation."

        # Save assistant message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

        with chat_container:
            st.chat_message("assistant", avatar=":material/gavel:").write(reply)

with tab2:
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

# with tab2:
#     generator = pipeline('text-generation', model='gpt2',framework='pt',from_tf=True)  # Or replace 'gpt2' with your model name
#     chat_container = st.container()

#     # Display history of the chat
#     for role, text in st.session_state.chat_sessions[st.session_state.active_chat]:
#         with chat_container.chat_message(role):
#             chat_container.write(text)

#     # Step-by-step question flow for generating documents
#     if "answers" not in st.session_state:
#         st.session_state.answers = {}

#     def ask_question_and_collect_answer(question):
#         user_input = st.text_input(question)
#         if user_input:
#             return user_input
#         return None

#     # Ask questions dynamically based on the prompt provided
#     user_prompt = st.text_input("What kind of legal document do you want to generate?")

#     if user_prompt:
#         # Use Hugging Face model to determine the document type and flow based on the prompt
#         response = generator(user_prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
#         st.session_state.chat_sessions[st.session_state.active_chat].append(('assistant', response))
#         st.write(response)  # Display LLM response

#         # Use Hugging Face model to ask additional questions (e.g., for a Khula petition)
#         questions = ["What is the wife's full name?", "What is the husband's full name?", "When did the marriage take place?"]
        
#         for question in questions:
#             if question not in st.session_state.answers:
#                 answer = ask_question_and_collect_answer(question)
#                 if answer:
#                     st.session_state.answers[question] = answer

#         # Once all questions are answered, generate the document
#         if len(st.session_state.answers) == len(questions):
#             draft = f"""
#             IN THE FAMILY COURT AT [City Name]

#             In the matter of:
#             {st.session_state.answers['What is the wifes full name?']} (Petitioner)
#             Versus
#             {st.session_state.answers['What is the husbands full name?']} (Respondent)

#             PETITION FOR KHULA

#             Respectfully Sheweth:

#             1. That the petitioner, {st.session_state.answers['What is the wifes full name?']}, was married to the respondent, {st.session_state.answers['What is the husbands full name?']}, on {st.session_state.answers['When did the marriage take place?']}.

#             PRAYER:
#             The petitioner respectfully prays for Khula (divorce) from the respondent.
            
#             Petitioner: {st.session_state.answers['What is the wifes full name?']}
#             Date: {datetime.date.today().strftime('%d %B %Y')}
#             """

#             # Display the generated draft
#             st.subheader("Generated Khula Petition")
#             st.code(draft, language="text")

#             # Function to create PDF
#             def generate_pdf(text, filename="Khula_Petition.pdf"):
#                 pdf = FPDF()
#                 pdf.set_auto_page_break(auto=True, margin=15)
#                 pdf.add_page()

#                 pdf.set_font("Arial", size=12)
#                 pdf.multi_cell(0, 10, text)

#                 pdf.output(filename)
#                 return filename

#             # Function to create DOCX
#             def generate_docx(text, filename="Khula_Petition.docx"):
#                 doc = Document()
#                 doc.add_paragraph(text)
#                 doc.save(filename)
#                 return filename

#             # Function to create TXT
#             def generate_txt(text, filename="Khula_Petition.txt"):
#                 with open(filename, "w") as file:
#                     file.write(text)
#                 return filename

#             # Generate PDF, DOCX, and TXT files and provide download options
#             pdf_filename = generate_pdf(draft)
#             docx_filename = generate_docx(draft)
#             txt_filename = generate_txt(draft)

#             # Provide the download options
#             with open(pdf_filename, "rb") as f:
#                 st.download_button("⬇️ Download PDF", f, file_name=pdf_filename)

#             with open(docx_filename, "rb") as f:
#                 st.download_button("⬇️ Download DOCX", f, file_name=docx_filename)

#             with open(txt_filename, "rb") as f:
#                 st.download_button("⬇️ Download TXT", f, file_name=txt_filename)








