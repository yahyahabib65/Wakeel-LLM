import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

# Optional: specify tesseract path if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_pdf_images(pdf_path, lang='eng'):
    # Open the PDF
    doc = fitz.open(pdf_path)
    extracted_text = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        print(f"Page {page_index + 1}: {len(image_list)} images")

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Load image with PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Apply OCR
            text = pytesseract.image_to_string(image, lang=lang)

            print(f"  Image {img_index + 1}: Extracted {len(text)} characters")
            extracted_text.append(text)

    doc.close()
    return "\n".join(extracted_text)

# Example usage
pdf_file = 'raw_data/image_pdf/family_law_manual.pdf'
output_text = extract_text_from_pdf_images(pdf_file)

# Save to text file
with open("processed_data/family_law_manual.txt", "w", encoding="utf-8") as f:
    f.write(output_text)

print("Text extraction complete. Output saved to 'extracted_text.txt'.")
