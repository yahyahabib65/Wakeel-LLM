
<p align="center">
  <img width="385" alt="Screenshot 2025-04-20 at 3 31 46 PM" src="https://github.com/user-attachments/assets/d2e5e2ed-4e7d-44de-a1e0-903d73c4e5ff" />
</p>
Wakeel is an AI-powered legal assistant that focuses on Pakistan Family Law, supporting Urdu and English interaction. Tailored for Pakistan’s context, it enables Urdu-speaking users and lawyers to access legal advice, draft petitions, and retrieve relevant case laws through LLM-driven consultation and citation tools.

<p align="center">
  <img width="909" alt="Screenshot 2025-04-20 at 3 34 00 PM" src="https://github.com/user-attachments/assets/b074f91f-0ca1-4a06-9bb7-2f4df39efb70" />
</p>

## Features  
*As of 13 April 2025*

### 1. Tab 1: Legal Consulting  
Designed for both general users and legal professionals.  
- **Input**: Text (PDF upload/Speech Input planned; demo supports text only)  
- **Method**: Fine-tuned LLM and RAG-based model  
- **Output**: A concise, easy-to-understand summary or explanation of the input document or question. 

### 2. Tab 2: Drafting  
Tailored for lawyers to generate legal drafts, such as a *Khula Petition*.  
- **Input**: Text (User data needed to complete the Draft)
- **Method**: Agentic AI and prompt-based generation  
- **Output**: Draft text file customized to the case (Demo: Khula Petition)  

### 3. Tab 3: Citations  
Designed for lawyers to automatically retrieve relevant legal precedents based on user input.  
- **Input**: Text  
- **Method**: RAG-based citation retrieval  
- **Output**: Top 3 matching case laws with summaries and source links  


## Data Sources

https://www.pakistanlawsite.com/Login/Check
https://www.peshawarhighcourt.gov.pk/PHCCMS/reportedJudgments.php?action=search

## Getting Started

1. Clone the repository.
2. Install Python and its dependencies used in project
3. use streamlit to run project-llm.py
