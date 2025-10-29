from typing import List
from pathlib import Path
import re

from huggingface_hub import snapshot_download
from langchain_experimental import text_splitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------
# Token utilities
# --------------------
def _get_tokenizer():
    """
    Prefer tiktoken cl100k_base. Fallback to whitespace split.
    Returns a callable that maps text -> list[int|str].
    """
    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: enc.encode(s)
    except Exception:
        return lambda s: s.split()


_TOKENIZE = _get_tokenizer()


def count_tokens(text: str) -> int:
    return len(_TOKENIZE(text))


def split_by_tokens(
        text: str,
        max_tokens: int,
        min_tokens: int = 100,
        overlap: int = 50,
) -> List[str]:
    """
    Split text into slices by token count. Never cross section boundaries if you
    call it per-section. Keeps overlap tokens between slices. Drops sub-100 token
    fragments unless the whole text is < min_tokens.
    """
    toks = _TOKENIZE(text)
    n = len(toks)
    if min_tokens <= n <= max_tokens:
        return [text]

    out = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        piece = toks[i:j]
        if len(piece) >= min_tokens or (i == 0 and j == n):
            out.append(piece)
        # back off by overlap for the next window
        nxt = j - overlap
        i = nxt if nxt > i else j

    # detokenize
    if isinstance(toks[0] if toks else "", str):
        # fallback path used whitespace split; join back
        return [" ".join(p) for p in out]

    try:
        import tiktoken  # type: ignore
        enc = tiktoken.get_encoding("cl100k_base")
        return [enc.decode(p) for p in out]
    except Exception:
        # ultimate fallback
        return [" ".join(p) for p in out]


# --------------------
# Furniture stripping
# --------------------
# Remove page numbers, headers/footers, and common PDF artifacts
FURNITURE_INLINE = (
    r"(?:^\s*\d+\s*$)"                # bare page numbers
    r"|(?:^Author Manuscript.*$)"     # 'Author Manuscript' banners
    r"|(?:^bioRxiv.*$)"               # preprint headers
    r"|(?:^\s*Page\s+\d+.*$)"         # 'Page N' lines
    r"|(?:^\s*Figure\s+\d+.*$)"       # standalone figure headers
    r"|(?:^\s*Table\s+\d+.*$)"        # standalone table headers
)


def strip_furniture(text: str) -> str:
    lines = [
        ln for ln in text.splitlines()
        if not re.match(FURNITURE_INLINE, ln.strip(), flags=re.IGNORECASE)
    ]
    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


# --------------------
# Fixed-size chunker
# --------------------
def getFixedChunker(chunk_size: int, chunkCountSymbol: str = " ") -> CharacterTextSplitter:
    """
    Character-based splitter with small overlap. Use only for quick tests.
    """
    return CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.08),
        separator=chunkCountSymbol,
        strip_whitespace=False,
    )


# --------------------
# Semantic chunker model (LangChain SemanticChunker)
# --------------------
def loadModel(
        modelName: str,
        modelPath: str,
        minChunkSize: int = 400,
) -> text_splitter.SemanticChunker:
    """
    Returns a LangChain SemanticChunker configured with HF embeddings.
    """
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    hf = HuggingFaceEmbeddings(
        model_name=(modelPath + modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    # percentile thresholding; buffer_size=1 for tighter boundaries
    return text_splitter.SemanticChunker(
        embeddings=hf,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=93,
        min_chunk_size=minChunkSize,
    )


def downloadModel(modelName: str, folderPath: str = "./models/") -> None:
    """
    Lazy local cache of HF model snapshot.
    """
    target = Path(folderPath) / modelName
    if not target.is_dir():
        snapshot_download(repo_id=modelName, local_dir=str(target))


def getModel(modelName: str, minChunkSize: int = 400) -> text_splitter.SemanticChunker:
    downloadModel(modelName)
    return loadModel(modelName=modelName, modelPath="./models/", minChunkSize=minChunkSize)


# --------------------
# Embeddings handle (if needed elsewhere, e.g., Qdrant)
# --------------------
def getEmbeddings(modelPath: str, modelName: str) -> HuggingFaceEmbeddings:
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": False}
    return HuggingFaceEmbeddings(
        model_name=(modelPath + modelName),
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )


# --------------------
# High-level helpers
# --------------------
def splitText(splitterModel, text: str) -> List[str]:
    """
    Pass-through to LangChain splitter. Use strip_furniture before calling this
    if the source may contain headers/footers.
    """
    cleaned = strip_furniture(text)
    return splitterModel.split_text(cleaned)


# --------------------
# Local demo (optional)
# --------------------
if __name__ == "__main__":
    # Minimal self-test. Safe to remove.
    sample = (
            "Abstract\nA short abstract.\n\n"
            "Introduction\nThis is a long paragraph that should be token-split if it exceeds the "
            "configured threshold. " * 40
            + "\n\nMethods\nStep 1. Step 2. Step 3.\n"
    )

    # Token-based split demo
    chunks = split_by_tokens(sample, max_tokens=200, min_tokens=100, overlap=40)
    print(f"[token-split] produced {len(chunks)} chunks:",
          [count_tokens(c) for c in chunks])

    # Semantic split demo
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    downloadModel(model_name)
    sem = loadModel(modelName=model_name, modelPath="./models/", minChunkSize=200)
    sem_chunks = splitText(sem, sample)
    print(f"[semantic-split] produced {len(sem_chunks)} chunks:",
          [len(c) for c in sem_chunks])
