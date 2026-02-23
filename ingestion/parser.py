import io
import json
import re
import fitz
from bs4 import BeautifulSoup
from docx import Document


class Parser:

    # =========================
    # TEXT CLEANER (NEW)
    # =========================
    def clean_text(self, text: str) -> str:
        """
        Clean noisy extracted text:
        - Remove ASCII table borders
        - Fix broken medication lines
        - Normalize whitespace
        """

        # Remove table border characters
        text = re.sub(r"\|+", " ", text)
        text = re.sub(r"\+[-]+\+", " ", text)

        # Fix broken dosage lines (e.g., "20...\n mg")
        text = re.sub(r"\.\.\.\s*\n\s*mg", "mg", text)

        # Join split lines like:
        # "Lisinopril 20...\nmg"
        text = re.sub(r"(\w+)\s*\n\s*(mg|g|ml)", r"\1 \2", text)

        # Remove excessive newlines
        text = re.sub(r"\n{2,}", "\n", text)

        # Normalize spaces
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    # =========================
    # PDF
    # =========================
    def parse_pdf(self, file_bytes):
        text = ""
        pdf_stream = io.BytesIO(file_bytes)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        for page in doc:
            text += page.get_text()

        return self.clean_text(text)

    # =========================
    # JSON
    # =========================
    def parse_json(self, file_bytes):
        data = json.loads(file_bytes.decode("utf-8"))
        text = "\n".join([f"{k}: {v}" for k, v in data.items()])
        return self.clean_text(text)

    # =========================
    # DOCX
    # =========================
    def parse_docx(self, file_bytes):
        text = ""
        doc = Document(io.BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"

        return self.clean_text(text)

    # =========================
    # HTML
    # =========================
    def parse_html(self, file_bytes):
        soup = BeautifulSoup(file_bytes, "html.parser")
        text = soup.get_text(separator="\n")
        return self.clean_text(text)

    # =========================
    # Router
    # =========================
    def parse(self, file_name, file_bytes):

        if file_name.endswith(".pdf"):
            return self.parse_pdf(file_bytes)

        elif file_name.endswith(".json"):
            return self.parse_json(file_bytes)

        elif file_name.endswith(".docx"):
            return self.parse_docx(file_bytes)

        elif file_name.endswith(".html"):
            return self.parse_html(file_bytes)

        else:
            raise ValueError("Unsupported file type")