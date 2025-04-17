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
                st.download_button("📥 Download PDF", chat["content"], file_name="legal_draft.pdf")

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

        # Mock AI Response
        ai_response = f"Here’s your legal consulting answer for: *{user_input}*"
        st.session_state.chat_history.append({
            "role": "ai",
            "content": ai_response,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        st.experimental_rerun()

# Tab 2: Draft Generator
with tab2:
    st.title("📄 Draft a Legal Document")
    draft_input = st.text_area("Describe the document you want to generate", height=150)
    generate = st.button("Generate Draft")

    if generate and draft_input:
        # Mock Draft Output
        draft_output = f"**Draft for:** {draft_input}\n\nThis is your AI-generated legal document."
        st.markdown(draft_output)
        st.download_button("📥 Download Draft", draft_output, file_name="legal_draft.txt")
