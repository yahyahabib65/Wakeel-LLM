<p align="center">
  <img width="385" alt="Screenshot 2025-04-20 at 3 31 46 PM" src="https://github.com/user-attachments/assets/d2e5e2ed-4e7d-44de-a1e0-903d73c4e5ff" />
</p>

# Wakeel LLM: AI-Powered Legal Assistant for Pakistan Family Law

Wakeel LLM is an AI-powered legal assistant focused on Pakistan Family Law, supporting both Urdu and English. It empowers users and lawyers to access legal advice, draft petitions, and retrieve relevant case laws through advanced LLM-driven consultation and citation tools. Wakeel is tailored for the Pakistani legal context, making legal information more accessible and actionable.

<p align="center">
  <img width="909" alt="Screenshot 2025-04-20 at 3 34 00 PM" src="https://github.com/user-attachments/assets/b074f91f-0ca1-4a06-9bb7-2f4df39efb70" />
</p>

---

## Features  
*As of 13 April 2025*

### 1. Legal Consulting  
Designed for both general users and legal professionals.  
- **Input**: Text (PDF upload/Speech Input planned; demo supports text only)  
- **Method**: Fine-tuned LLM and RAG-based model  
- **Output**: A concise, easy-to-understand summary or explanation of the input document or question. 

### 2. Drafting  
Tailored for lawyers to generate legal drafts, such as a *Khula Petition*.  
- **Input**: Text (User data needed to complete the Draft)  
- **Method**: Agentic AI and prompt-based generation  
- **Output**: Draft text file customized to the case (Demo: Khula Petition)  

### 3. Citations  
Designed for lawyers to automatically retrieve relevant legal precedents based on user input.  
- **Input**: Text  
- **Method**: RAG-based citation retrieval  
- **Output**: Top 3 matching case laws with summaries and source links  

---

## Data Sources

- https://www.pakistanlawsite.com/Login/Check  
- https://www.peshawarhighcourt.gov.pk/PHCCMS/reportedJudgments.php?action=search  

---

## Getting Started

### Prerequisites

- Python 3.11+
- [pip](https://pip.pypa.io/en/stable/)
- [Streamlit](https://streamlit.io/)
- (Optional) CUDA-enabled GPU for faster inference

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourusername/Wakeel-LLM.git
    cd Wakeel-LLM
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```
    If you are using GPU, ensure you have the correct versions of PyTorch and CUDA installed.

3. **Download or place your models:**
    - Place your fine-tuned LLM and LoRA adapters in the `models/` directory as required.
    - (Optional) Download additional resources as needed.

### Running the Application

To launch the main Streamlit app:

```sh
streamlit run Project-LLM.py
```

- The app will open in your browser at `http://localhost:8501/` by default.
- Follow the on-screen instructions to use the Legal Consulting, Drafting, and Citations tabs.

### Deployment

For production or server deployment:

1. **Set up a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2. **Install dependencies as above.**

3. **Run Streamlit with appropriate host/port:**
    ```sh
    streamlit run Project-LLM.py --server.port 8501 --server.address 0.0.0.0
    ```

4. **(Optional) Use a process manager** (e.g., `pm2`, `systemd`, or `supervisor`) to keep the app running.

5. **(Optional) Set up a reverse proxy** (e.g., Nginx) for HTTPS and domain routing.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or support, please contact [your-email@example.com](mailto:your-email@example.com).