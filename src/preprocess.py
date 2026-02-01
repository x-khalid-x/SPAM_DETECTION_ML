import re

def normalize_spam_patterns(text: str) -> str:
    """
    Normalisation "spam-aware" :
    - URL -> __URL__
    - EMAIL -> __EMAIL__
    - NUMBER -> __NUMBER__
    - MONEY -> __MONEY__
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " __URL__ ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " __EMAIL__ ", text)
    text = re.sub(r"\b\d+(\.\d+)?\b", " __NUMBER__ ", text)
    text = re.sub(r"[$€£]\s*\d+", " __MONEY__ ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
