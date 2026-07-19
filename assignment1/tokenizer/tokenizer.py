from cs336_basics.tokenizer.train_bpe import split_keep_special
from cs336_basics.common.constants import ENCODE_FORMAT

from itertools import pairwise
from typing import Iterable, Iterator

import json

class Tokenizer:

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.format = 'utf-8'
        self.vocab = vocab 
        self.token_to_id = {token: id for id, token in self.vocab.items()}
        self.merges = merges
        self.merges_with_rank = {pair : i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens
        if special_tokens:
            self._append_vocab(vocab, special_tokens)
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, 'r', encoding=ENCODE_FORMAT) as file: 
            raw_vocab = json.load(file)
            vocab = {int(k): v.encode(ENCODE_FORMAT) for k, v in raw_vocab.items()}

        merges = []
        with open(merges_filepath, 'r', encoding=ENCODE_FORMAT) as f:
            for line in f:
                line = line.strip()
                # 跳过空行或注释（如 #version: 0.2）
                if not line or line.startswith('#'):
                    continue
                    
                # 假设每行是以空格分隔的两个 token 字符串，例如： "e r"
                parts = line.split()
                if len(parts) == 2:
                    merges.append((parts[0].encode(ENCODE_FORMAT), parts[1].encode(ENCODE_FORMAT)))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        encoded_seqs = []

        # split by special token 
        for is_special, piece in split_keep_special(text, self.special_tokens):
            if is_special:
                encoded_seqs.append(self.token_to_id.get(piece.encode(ENCODE_FORMAT)))
            else:
                # Normal seqs needs to encode by the order of merge
                encoded_tokens = self._encode_with_rank(piece)
                encoded_seqs.extend(self.token_to_id.get(token) for token in encoded_tokens)

        return encoded_seqs
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        for text in iterable: 
            for is_special, piece in split_keep_special(text, self.special_tokens):
                if is_special:
                    yield self.token_to_id.get(piece.encode(ENCODE_FORMAT))
                else:
                    # Normal seqs needs to encode by the order of merge
                    encoded_tokens = self._encode_with_rank(piece)
                    yield from (self.token_to_id.get(token) for token in encoded_tokens)
    
    def decode(self, ids: list[int]) -> str:
        decode_tokens = [self.vocba.get(id) for id in ids]
        text_bytes = b''.join(decode_tokens)

        return text_bytes.decode(ENCODE_FORMAT, errors='replace')
    
    def _encode_with_rank(self, piece: str):
        encoded_tokens = piece.encode(ENCODE_FORMAT)
        while len(encoded_tokens) > 2: 
            # for pair in pairwise(encoded_text):
            pairs = set(pairwise(encoded_tokens))

            best_pair = min(pairs, lambda p : self.merges_with_rank.get(p, float('inf')))
            if best_pair not in self.merges_with_rank:
                break
            new_tokens = [] 
            i = 0 
            while i < len(encoded_tokens):
                if i < len(encoded_tokens) - 1 and encoded_tokens[i] == best_pair[0] and encoded_tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2 
                else: 
                    new_tokens.append(encoded_tokens[i])
                    i += 1 
            encoded_tokens = new_tokens
        return encoded_tokens





    def _append_vocab(self, vocab: dict[int, bytes], special_tokens: list[str]):
        current_max_key = max(vocab.keys(), default=0) 
        existing_values = set(vocab.values())
        for special_token in special_tokens:
            encoded_token = special_token.encode(self.format)
            if encoded_token not in existing_values:
                current_max_key += 1 
                vocab[current_max_key] = encoded_token
                existing_values.add(encoded_token)
