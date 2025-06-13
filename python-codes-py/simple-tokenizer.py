class SimpleTokenizer:
    """
    Simple character-level tokenizer for demonstration
    """
    def __init__(self, text):
        self.chars = sorted(list(set(text.lower())))
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        
        # Special tokens
        self.pad_token = 0
        self.sos_token = 1  # Start of sequence
        self.eos_token = 2  # End of sequence
        
    def encode(self, text):
        """Convert text to token indices"""
        return [self.char_to_idx.get(c, 0) for c in text.lower()]
    
    def decode(self, tokens):
        """Convert token indices back to text"""
        return ''.join([self.idx_to_char.get(t, '') for t in tokens])
    
    def encode_with_special_tokens(self, text):
        """Encode text with start and end tokens"""
        tokens = [self.sos_token] + self.encode(text) + [self.eos_token]
        return tokens

# Example of tokenization
sample_text = "hello world this is a transformer model"
tokenizer = SimpleTokenizer(sample_text)

print("Vocabulary:", tokenizer.chars)
print("Vocab size:", tokenizer.vocab_size)
print("Original text:", sample_text)
print("Encoded:", tokenizer.encode(sample_text))
print("Decoded:", tokenizer.decode(tokenizer.encode(sample_text)))