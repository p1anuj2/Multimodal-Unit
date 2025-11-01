from transformers import AutoTokenizer

def get_tokenizer(name: str = "bert-base-uncased"):
    """
    Centralizes tokenizer choice. Adjust to match your text encoder if needed.
    """
    return AutoTokenizer.from_pretrained(name)
