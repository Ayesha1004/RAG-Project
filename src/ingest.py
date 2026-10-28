from pathlib import Path
from pypdf import PdfReader


def load_documents(folder="documents"):
    documents = []

    for file_path in Path(folder).glob("*.pdf"):
        reader = PdfReader(file_path)

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                documents.append({
                    "text": text,
                    "source": file_path.name,
                    "page": page_number
                })

    return documents