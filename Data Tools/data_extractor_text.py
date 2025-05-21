from pypdf import PdfReader
import os

# Folder containing PDFs
pdf_folder = "raw_data/text_pdf"
output_folder = "processed_data"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all PDF files
for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"Saved: {txt_path}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
