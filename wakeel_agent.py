import streamlit as st
from docx import Document
import datetime

st.set_page_config(page_title="Legal Draft Chatbot", page_icon="📜")
st.title("📜 Muslim Family Law Drafting Assistant")

# ---------------------------------------
# Supported Draft Types and Questions
# ---------------------------------------
draft_options = {
    "Khula Petition": [
        ("wife_name", "👩 What is the wife's full name?"),
        ("husband_name", "🧔 What is the husband's full name?"),
        ("marriage_date", "📅 When did the marriage take place?"),
        ("place_of_marriage", "📍 Where was the marriage held?"),
        ("reason_for_khula", "💔 What is the reason for seeking Khula?"),
        ("mehr_details", "💰 What were the Mehr details?"),
        ("children_details", "👶 Are there any children?"),
        ("prayer", "🙏 What relief is being sought?")
    ],
    "Property Transfer After Death": [
        ("deceased_name", "🪦 What is the full name of the deceased?"),
        ("date_of_death", "📅 When did the person pass away?"),
        ("property_details", "🏠 Describe the property to be transferred."),
        ("legal_heirs", "👪 List the legal heirs and their shares."),
        ("transfer_reason", "⚖️ State the reason for the transfer."),
    ],
    "Will Deed": [
        ("testator_name", "👤 Full name of the person making the will:"),
        ("testator_address", "🏠 Address of the testator:"),
        ("property_to_be_distributed", "🏘️ Describe the property to be included in the will:"),
        ("beneficiaries", "👨‍👩‍👧‍👦 List the names and shares of beneficiaries:"),
        ("executor_name", "📜 Who will execute this will?")
    ],
    "Marriage Registration": [
        ("bride_name", "👰 Full name of the bride:"),
        ("groom_name", "🤵 Full name of the groom:"),
        ("marriage_date", "📅 Date of marriage:"),
        ("place_of_marriage", "📍 Place of marriage:"),
        ("witnesses", "🧾 Names of two witnesses:"),
        ("nikah_registrar", "📋 Name of the Nikah Registrar:")
    ]
}

# ---------------------------------------
# Draft Generators
# ---------------------------------------

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

# ---------------------------------------
# Export Function
# ---------------------------------------
def export_to_docx(text, filename="Legal_Draft.docx"):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    return filename

# ---------------------------------------
# Session State
# ---------------------------------------
if "draft_type" not in st.session_state:
    st.session_state.draft_type = None
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.finished = False
    st.session_state.chat_history = []

# ---------------------------------------
# Draft Type Selector (only first time)
# ---------------------------------------
if not st.session_state.draft_type:
    st.session_state.draft_type = st.selectbox("📑 Select the type of legal draft you want to generate:", list(draft_options.keys()))
    # st.rerun()

# ---------------------------------------
# Chat-Like Q&A
# ---------------------------------------
questions = draft_options[st.session_state.draft_type]

# Show chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# Ask next question
if not st.session_state.finished:
    current_field, question = questions[st.session_state.step]
    with st.chat_message("assistant"):
        st.markdown(question)

    user_input = st.chat_input("Your answer...")

    if user_input:
        st.session_state.answers[current_field] = user_input
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", question))

        st.session_state.step += 1

        if st.session_state.step >= len(questions):
            st.session_state.finished = True
            draft = generate_draft(st.session_state.draft_type, st.session_state.answers)
            filename = export_to_docx(draft, f"{st.session_state.draft_type.replace(' ', '_')}.docx")

            with st.chat_message("assistant"):
                st.success("✅ Your draft is ready!")
                st.code(draft)
                with open(filename, "rb") as f:
                    st.download_button("⬇️ Download Draft", f, file_name=filename)
                    st.download_button("⬇️ Download Draft (PDF)", f, file_name=filename)

        # st.rerun()
