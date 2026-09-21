
import os
import json
import uuid

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# -----------------------------
# Configuration
# -----------------------------

DATA_FOLDER = "daraz_knowledge_base"
INDEX_FOLDER = "faiss_index"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk settings
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


# -----------------------------
# Create folders
# -----------------------------

os.makedirs(INDEX_FOLDER, exist_ok=True)


# -----------------------------
# Extract text from PDF
# -----------------------------

def extract_pdf_text(pdf_path):
    """
    Read a PDF and return all extracted text.
    """

    reader = PdfReader(pdf_path)

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


# -----------------------------
# Split text into chunks
# -----------------------------

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.
    """

    text = text.replace("\n", " ")
    text = " ".join(text.split())

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start = end - overlap

    return chunks


# -----------------------------
# Read all PDFs
# -----------------------------

documents = []

print("Reading PDF documents...\n")

for root, dirs, files in os.walk(DATA_FOLDER):

    for filename in files:

        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(root, filename)

            # Department = immediate folder containing PDF
            department = os.path.basename(root)

            print("Reading:", pdf_path)

            try:

                text = extract_pdf_text(pdf_path)

                if not text.strip():
                    print("  WARNING: No text found")
                    continue

                chunks = split_text(text)

                for chunk_number, chunk in enumerate(chunks):

                    metadata = {
                        "id": str(uuid.uuid4()),
                        "department": department,
                        "source_file": filename,
                        "chunk_number": chunk_number,
                        "text": chunk
                    }

                    documents.append(metadata)

            except Exception as e:

                print("  ERROR:", e)


print("\nTotal chunks created:", len(documents))


# -----------------------------
# Check documents
# -----------------------------

if len(documents) == 0:

    raise ValueError(
        "No PDF chunks were created. "
        "Check your daraz_knowledge_base folder."
    )


# -----------------------------
# Load embedding model
# -----------------------------

print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# -----------------------------
# Create embeddings
# -----------------------------

texts = [doc["text"] for doc in documents]

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)


# -----------------------------
# Create FAISS index
# -----------------------------

embedding_dimension = embeddings.shape[1]

print("\nEmbedding dimension:", embedding_dimension)

index = faiss.IndexFlatIP(embedding_dimension)

index.add(embeddings)


# -----------------------------
# Save FAISS index
# -----------------------------

index_path = os.path.join(
    INDEX_FOLDER,
    "index.faiss"
)

faiss.write_index(index, index_path)


# -----------------------------
# Save metadata
# -----------------------------

metadata_path = os.path.join(
    INDEX_FOLDER,
    "metadata.json"
)

with open(metadata_path, "w", encoding="utf-8") as f:

    json.dump(
        documents,
        f,
        ensure_ascii=False,
        indent=2
    )


# -----------------------------
# Finished
# -----------------------------

print("\n================================")
print("INGESTION COMPLETED")
print("================================")

print("Documents/chunks:", len(documents))
print("FAISS index:", index_path)
print("Metadata:", metadata_path)
