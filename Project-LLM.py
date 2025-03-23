import streamlit as st
from datetime import datetime

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

# Main Window - Chat Display
st.title("💬 Wakeel - Your AI Legal Assistant")
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
            st.download_button("📥 Download PDF", chat["content"], file_name="legal_draft.pdf")

st.write("---")

# Bottom Input Box
st.subheader("Ask Your Question")
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    uploaded_file = st.file_uploader("📎", label_visibility="collapsed")
with col2:
    user_input = st.text_area("Type your query", label_visibility="collapsed", height=70)
with col3:
    voice_input = st.button("🎤")

send = st.button("Send")

# Handle Input
if send and user_input:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    # Mock AI Response for now
    ai_response = f"Here’s your legal draft answer for: *{user_input}* (This will be generated from the model)"
    st.session_state.chat_history.append({
        "role": "ai",
        "content": ai_response,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    st.experimental_rerun()
