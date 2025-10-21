from dataclasses import dataclass
import os

@dataclass(frozen=True)
class NCBIConfig:
    email: str
    api_key: str
    base: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def load_config() -> NCBIConfig:
    email = os.getenv("NCBI_EMAIL", "").strip()
    api_key = os.getenv("NCBI_API_KEY", "").strip()
    if not email or not api_key:
        raise RuntimeError(
            "NCBI_EMAIL and NCBI_API_KEY must be set (see .env). "
            "Do NOT commit .env to git."
        )
    return NCBIConfig(email=email, api_key=api_key)
