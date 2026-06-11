import json
import uuid
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from app.config import Settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt", ".json"}

ARABIC_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "cp1256", "iso-8859-6")


def read_text_file(file_path: Path) -> str:
    raw = file_path.read_bytes()
    for encoding in ARABIC_TEXT_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode text file: {file_path.name}")


def json_to_text(obj, indent: int = 0) -> str:
    text = ""
    prefix = "  " * indent
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, (dict, list)):
                text += f"{prefix}{key}:\n{json_to_text(value, indent + 1)}"
            else:
                text += f"{prefix}{key}: {value}\n"
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                text += f"{prefix}Item {index + 1}:\n{json_to_text(item, indent + 1)}"
            else:
                text += f"{prefix}Item {index + 1}: {item}\n"
    else:
        text += f"{prefix}{obj}\n"
    return text


def load_json_document(file_path: Path) -> list[Document]:
    data = json.loads(read_text_file(file_path))
    text = json_to_text(data).strip()
    return [Document(page_content=text, metadata={"source": file_path.name})]


def load_document(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        return loader.load()
    if suffix in {".docx", ".doc"}:
        loader = Docx2txtLoader(str(file_path))
        return loader.load()
    if suffix == ".json":
        return load_json_document(file_path)
    if suffix in {".md", ".txt"}:
        text = read_text_file(file_path)
        return [Document(page_content=text, metadata={"source": file_path.name})]

    raise ValueError(f"Unsupported file type: {suffix}")


def _pre_split_markdown(documents: list[Document], filename: str) -> list[Document]:
    if Path(filename).suffix.lower() != ".md":
        return documents

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "section"),
            ("##", "section"),
            ("###", "section"),
        ],
        strip_headers=False,
    )
    sectioned: list[Document] = []
    for doc in documents:
        sectioned.extend(header_splitter.split_text(doc.page_content))
    return sectioned or documents


def split_documents(
    documents: list[Document],
    settings: Settings,
    document_id: str,
    filename: str,
) -> list[Document]:
    tokenizer = AutoTokenizer.from_pretrained(settings.embedding_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    source_docs = _pre_split_markdown(documents, filename)
    chunks = splitter.split_documents(source_docs)
    enriched: list[Document] = []

    for index, chunk in enumerate(chunks):
        page = chunk.metadata.get("page")
        if page is not None:
            page = int(page) + 1

        section = chunk.metadata.get("section")
        if section and section not in chunk.page_content[:200]:
            chunk.page_content = f"{section}\n\n{chunk.page_content}"

        chunk_id = str(uuid.uuid4())
        chunk.metadata.update(
            {
                "document_id": document_id,
                "filename": filename,
                "chunk_index": index,
                "page": page,
                "source": filename,
                "section": section,
            }
        )
        chunk.id = chunk_id
        enriched.append(chunk)

    return enriched
