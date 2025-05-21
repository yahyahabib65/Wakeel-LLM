#region import libraries
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fpdf import FPDF
from docx import Document

#endregion
st.set_page_config(page_title="Legal Draft Generator", layout="wide")

#region model loader and response generator
@st.cache_resource
def load_model():
    base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Frontend\\tinyllama_lora_muslim_family_law")
    model = PeftModel.from_pretrained(base_model, "Frontend\\tinyllama_lora_muslim_family_law")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()


def generate_response(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
#endregion

#region streamlit UI
st.title("📄 Legal Draft Generator – Muslim Family Law")
st.write("Answer questions step-by-step to generate a legal document like Khula, Marriage, or Inheritance.")

if "draft_conversation" not in st.session_state:
    st.session_state.draft_conversation = []
if "draft_complete" not in st.session_state:
    st.session_state.draft_complete = False

# Start conversation
if not st.session_state.draft_conversation and not st.session_state.draft_complete:
    st.session_state.draft_conversation.append(("assistant", "What type of legal document would you like to generate? (e.g., Khula, Marriage, Inheritance)"))

# Display conversation
for role, text in st.session_state.draft_conversation:
    with st.chat_message(role):
        st.write(text)

# Continue conversation
if not st.session_state.draft_complete:
    user_input = st.chat_input(placeholder="Your answer...")
    if user_input:
        st.session_state.draft_conversation.append(("user", user_input))
        with st.spinner("Processing..."):
            try:
                user_responses = "\n".join([
                    f"User: {text}" for role, text in st.session_state.draft_conversation if role == "user"])
                prompt = (
                    "You are a legal assistant drafting a legal document under Pakistani Muslim Family Law.\n"
    "You must ask the user for only one specific piece of information at a time.\n"
    "When you have all required details, generate the full legal document.\n\n"
    f"Conversation so far:\n{chat_history}\n\n"
    "Now, either ask the next relevant question OR generate the legal document.\n"
    "Only ask one follow-up question at a time unless you have enough details to generate a draft. Do not repeat previous questions."
                )
                response = generate_response(prompt)
                if "IN THE FAMILY COURT" in response or "PETITION" in response or "DATED" in response:
                    st.session_state.draft_conversation.append(("assistant", response))
                    st.session_state.draft_complete = True
                else:
                    st.session_state.draft_conversation.append(("assistant", response))
            except Exception as e:
                st.error(f"Error: {e}")

# Show final draft and download
if st.session_state.draft_complete:
    draft = st.session_state.draft_conversation[-1][1]
    st.subheader("Generated Draft")
    st.code(draft, language="text")

    def generate_pdf(text, filename="Legal_Draft.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, text)
        pdf.output(filename)
        return filename

    def generate_docx(text, filename="Legal_Draft.docx"):
        doc = Document()
        doc.add_paragraph(text)
        doc.save(filename)
        return filename

    def generate_txt(text, filename="Legal_Draft.txt"):
        with open(filename, "w") as file:
            file.write(text)
        return filename

    pdf_filename = generate_pdf(draft)
    docx_filename = generate_docx(draft)
    txt_filename = generate_txt(draft)

    with open(pdf_filename, "rb") as f:
        st.download_button("⬇️ Download PDF", f, file_name=pdf_filename)
    with open(docx_filename, "rb") as f:
        st.download_button("⬇️ Download DOCX", f, file_name=docx_filename)
    with open(txt_filename, "rb") as f:
        st.download_button("⬇️ Download TXT", f, file_name=txt_filename)

    if st.button("🔁 Start Over"):
        st.session_state.draft_conversation = []
        st.session_state.draft_complete = False
        st.rerun()
#endregion
