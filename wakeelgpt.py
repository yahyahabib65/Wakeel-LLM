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
import asyncio  # Add at top if not already

from docx import Document
import base64
import streamlit.components.v1 as components

import re
from langdetect import detect
from googletrans import Translator

def detect_target_language(user_prompt):
    if re.search(r"اردو میں جواب دیں|Urdu", user_prompt, re.IGNORECASE):
        return "ur"
    elif re.search(r"انگریزی میں جواب دیں|English", user_prompt, re.IGNORECASE):
        return "en"
    else:
        return "en"  # Default

# def generate_response_with_translation(user_prompt, use_rag=False):
#     target_lang = detect_target_language(user_prompt)
#     translator = Translator()
#     input_lang = detect(user_prompt)

#     # Translate input prompt if needed
#     prompt_to_use = user_prompt
#     if input_lang != target_lang:
#         prompt_to_use = translator.translate(user_prompt, dest=target_lang).text

#     # Call appropriate response generator
#     if use_rag:
#         vector_store = load_rag_model()
#         model_response = generate_rag_response(prompt_to_use, vector_store)
#     else:
#         model_response = generate_response(prompt_to_use)

#     # Translate response back to original language if needed
#     if detect(model_response) != input_lang:
#         final_response = translator.translate(model_response, dest=input_lang).text
#     else:
#         final_response = model_response

#     return final_response

async def generate_response_with_translation(user_prompt, use_rag=False):
    target_lang = detect_target_language(user_prompt)
    translator = Translator()
    input_lang = detect(user_prompt)

    # Translate input prompt if needed
    prompt_to_use = user_prompt
    if input_lang != target_lang:
        translated = await translator.translate(user_prompt, dest=target_lang)
        prompt_to_use = translated.text

    # Call appropriate response generator
    if use_rag:
        vector_store = load_rag_model()
        model_response = generate_rag_response(prompt_to_use, vector_store)
    else:
        model_response = generate_response(prompt_to_use)

    # Translate response back to original language if needed
    if detect(model_response) != input_lang:
        translated_response = await translator.translate(model_response, dest=input_lang)
        final_response = translated_response.text
    else:
        final_response = model_response

    return final_response


# #endregion
# def set_bg_from_local(image_file):
#     with open(image_file, "rb") as image:
#         encoded = base64.b64encode(image.read()).decode()

#     css = f"""
#     <style>
#     .stApp {{
#         background-image: url("data:image/png;base64,{encoded}");
#         background-size: cover;
#         background-position: center;
#         background-repeat: no-repeat;
#     }}
#     </style>
#     """
#     st.markdown(css, unsafe_allow_html=True)

# # Usage
# set_bg_from_local("demobackground.jpeg")

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

SYSTEM_PROMPT = """You are a legal expert trained in Pakistani family and civil law. Your role is to explain the answer in both clear English and simple Urdu so that it is understandable by both lawyers and the general public.

1. The tone should be:
   - Clear and professional (for law students)
   - Simple and respectful (for general users)
2. Structure your response in two parts:
   - **English Explanation**
   - ** اردو وضاحت**
3. Keep the total response under 250 words for each language, 500 for both.
4. Avoid legal jargon unless necessary. If used, explain it clearly.
If a query is unrelated to Pakistani family law, politely refuse to answer and remind the user of your domain restriction.
Focus areas include: divorce, child custody, maintenance (nafaqah), polygamy, nikah, dissolution of marriage, guardianship, inheritance under family law, and related topics.

Now generate a response that answers the user's question."""



def generate_response(prompt_text):
    full_prompt = SYSTEM_PROMPT + "\n\n" + prompt_text
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    # inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
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
    
    prompt = st.chat_input(placeholder="Your message", key=f"chat_input_{st.session_state.active_chat}",accept_file=False)
    if prompt:
        # Save user message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("user", prompt))

        with chat_container:
            st.chat_message("user").write(prompt)

        # Generate response
        # reply = generate_response(prompt)
        # reply = generate_response_with_translation(prompt, use_rag=False)
        reply = asyncio.run(generate_response_with_translation(prompt, use_rag=False))

        # reply = "This is a placeholder response. Please implement the actual model response generation."

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
        # reply = generate_rag_response(prompt, vector_store)
        reply = asyncio.run(generate_response_with_translation(prompt, use_rag=True))

        # reply = "This is a placeholder response. Please implement the actual RAG response generation."

        # Save assistant message
        st.session_state.chat_sessions[st.session_state.active_chat].append(("assistant", reply))

        with chat_container:
            st.chat_message("assistant", avatar=":material/gavel:").write(reply)

with tab2:
    def export_to_docx(text, filename="Legal_Draft.docx"):
        doc = Document()
        doc.add_paragraph(text)
        doc.save(filename)
        return filename


    def generate_draft(draft_type, data):
        if draft_type == "Khula Petition":
            return f"""IN THE FAMILY COURT AT [City Name]

    In the matter of:
    {data['wife_name']} (Petitioner)
    Versus
    {data['husband_name']} (Respondent)

    PETITION FOR KHULA

    Respectfully Sheweth:

    1. That the petitioner was married to the respondent on {data['marriage_date']} at {data['place_of_marriage']}.
    2. That the relationship has irretrievably broken down due to: {data['reason_for_khula']}.
    3. That the petitioner is willing to return the mehr amount of {data['mehr_details']}.
    4. {data['children_details']}

    PRAYER:
    {data['prayer']}

    Petitioner: {data['wife_name']}
    Date: {datetime.date.today().strftime('%d %B %Y')}
    """
        elif draft_type == "Property Transfer After Death":
            return f"""PROPERTY TRANSFER DECLARATION

    This is to certify that {data['deceased_name']} passed away on {data['date_of_death']}.

    The following property is to be transferred:
    {data['property_details']}

    Legal heirs entitled to the property are:
    {data['legal_heirs']}

    Reason for transfer:
    {data['transfer_reason']}

    Date: {datetime.date.today().strftime('%d %B %Y')}
    """
        elif draft_type == "Will Deed":
            return f"""WILL DEED

    I, {data['testator_name']}, residing at {data['testator_address']}, being of sound mind and disposing memory, do hereby declare this to be my last will and testament.

    The following property shall be distributed:
    {data['property_to_be_distributed']}

    Beneficiaries and their shares:
    {data['beneficiaries']}

    Executor of this Will:
    {data['executor_name']}

    Executed on this day: {datetime.date.today().strftime('%d %B %Y')}
    """
        elif draft_type == "Marriage Registration":
            return f"""APPLICATION FOR MARRIAGE REGISTRATION

    Bride: {data['bride_name']}
    Groom: {data['groom_name']}
    Date of Marriage: {data['marriage_date']}
    Place of Marriage: {data['place_of_marriage']}

    Witnesses:
    {data['witnesses']}

    Nikah Registrar: {data['nikah_registrar']}

    Submitted on: {datetime.date.today().strftime('%d %B %Y')}
    """
        else:
            return "❌ Draft type not recognized."


    # 1) Initialize session state
    if "draft_type" not in st.session_state:
        st.session_state.draft_type = None
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.finished = False
        st.session_state.chat_history = []

    # 2) Select draft type once
    if st.session_state.draft_type is None:
        st.session_state.draft_type = st.selectbox(
            "📑 Select the type of legal draft:",
            ["Khula Petition", "Property Transfer After Death", "Will Deed", "Marriage Registration"],
        )

    # 3) Question bank
    draft_options = {
      "Khula Petition": [
         ("wife_name",      "👩 What is the wife's full name?"),
         ("husband_name",   "🧔 What is the husband's full name?"),
         ("marriage_date", "📅 When did the marriage take place?"),
        ("place_of_marriage", "📍 Where was the marriage held?"),
        ("reason_for_khula", "💔 What is the reason for seeking Khula?"),
        ("mehr_details", "💰 What were the Mehr details?"),
        ("children_details", "👶 Are there any children?"),
        ("prayer", "🙏 What relief is being sought?")
      ],
      # other types …
    }

    # 4) Single container for chat history + input
    chat_area = st.container()

    # 5) Replay history
    for role, msg in st.session_state.chat_history:
        with chat_area:
            st.chat_message(role).markdown(msg)

    # 6) If not finished, ask next question
    if not st.session_state.finished:
        field, question = draft_options[st.session_state.draft_type][st.session_state.step]
        with chat_area:
            st.chat_message("assistant").markdown(question)

        # *** Give the input widget a stable key tied to the step ***
        user_input = chat_area.chat_input(
            "Your answer…",
            key=f"answer_{st.session_state.step}"
        )

        if user_input:
            # record
            st.session_state.answers[field] = user_input
            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", question))
            st.session_state.step += 1

            # check completion
            if st.session_state.step >= len(draft_options[st.session_state.draft_type]):
                st.session_state.finished = True

    # 7) Generate & show draft once complete
    if st.session_state.finished:
        draft = generate_draft(st.session_state.draft_type, st.session_state.answers)
        filename = export_to_docx(draft, f"{st.session_state.draft_type}.docx")

        with chat_area:
            st.success("✅ Your draft is ready!")
            st.code(draft)
            with open(filename, "rb") as f:
                st.download_button("⬇️ Download Draft", f, file_name=filename)

        # reset button
        if st.button("🔁 Create New Draft"):
            for k in ["draft_type","step","answers","finished","chat_history"]:
                del st.session_state[k]
            st.rerun()