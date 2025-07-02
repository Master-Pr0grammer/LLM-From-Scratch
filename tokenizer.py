class Tokenizer():
    def __init__(self, vocab:list[str]):
        self.vocab_list = sorted(vocab)
        
        self.txt_to_idx = {char: idx for idx, char in enumerate(self.vocab_list)}
        self.idx_to_txt = {idx: char for idx, char in enumerate(self.vocab_list)}

    def encode(self, text) -> list[int]:
        return [self.txt_to_idx[char] for char in text]

    def decode(self, tokens:list[int]):
        return [self.idx_to_txt[idx] for idx in tokens]