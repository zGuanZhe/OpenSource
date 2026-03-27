def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.replace('\n', ' ')
    text = " ".join(text.split())
    return text

def preprocess_data(raw_texts):
    """Preprocess a list of raw texts."""
    return [clean_text(t) for t in raw_texts if len(t.strip()) > 0]
