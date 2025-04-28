import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

#region models and helper functions with initialization
@st.cache_resource
def load_model():
    base_model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained("Frontend\\tinyllama_lora_muslim_family_law")

    model = PeftModel.from_pretrained(base_model, "Frontend\\tinyllama_lora_muslim_family_law")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# Helper function to generate model response
def generate_response(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply

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
        reply = generate_response(prompt)

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
    if prompt := st.chat_input(placeholder="Your message",accept_file=True,key="draft"):
        messages.chat_message("user").write(prompt.text)
        messages.chat_message("assistant",avatar=":material/gavel:").write(f"Echo: {prompt.text}")

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