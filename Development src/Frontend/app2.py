import streamlit as st
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import base64

# Page config
st.set_page_config(page_title="Wakeel - Legal Assistant", layout="wide")

# # Function to add a background image
# def add_background_image(image_path):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("{image_path}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Call the function with the relative path to your image
# add_background_image("D:\\LUMS\\Wakeel-LLM\\Frontend\\background.png") 
def set_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Usage
set_bg_from_local("D:\\LUMS\\Wakeel-LLM\\Frontend\\demobackground.jpeg")
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

# Tabs for Consulting and Legal Drafts
tab1, tab2 = st.tabs(["💼 Legal Consulting", "📄 Draft Generator"])

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
        uploaded_file = st.file_uploader("📎", label_visibility="collapsed", key="file_uploader_draft")
    with col2:
        user_input = st.text_area("Describe your draft requirements", label_visibility="collapsed", height=70)
    with col3:
        voice_input = st.button("🎤",key="voice_input_consulting")
    generate_button = st.button("Generate Draft", key="generate_draft")

    if generate_button and user_input:
        # Save the user's query to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        # Directly use the model to generate the draft
        combined_input = (
            "Please generate a legal draft based on the following input:\n\n"
            + user_input
        )

        # Pass the input to the LLM for generation
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

#         def add_background_image(image_path):
#     st.markdown(
#         f"""
#         <style>
#         .stApp {{
#             background-image: url("{image_path}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# # Call the function with the relative path to your image
# add_background_image("background.png")  # Replace with your image file path