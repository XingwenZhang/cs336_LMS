from itertools import pairwise
import regex as re
from collections import Counter
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def train_bpe(input_path : str, 
              vocab_size: int, 
              special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]: 

    # data validation 
    if vocab_size < 256 + len(special_tokens):
        raise("Invalid vocab size")

    
    # Return 
    # vocab:dict[tokenId, bytes]; merges:list[tuple[bytes, bytes]]
    ENCODE_FORMAT = 'utf-8'

    with open(input_path, 'r', encoding='utf-8') as file: 
        content = file.read() 
        
    normal_seqs = pre_tokenize_content(content, special_tokens)

    vocabs = {i : bytes([i]) for i in range(256)} # basic utf-8

    index = 256 

    for special_token in special_tokens:
        vocabs[index] = special_token.encode(ENCODE_FORMAT)
        index += 1
    

    available_size = vocab_size - index
    
    merges = merge(normal_seqs, available_size, index, vocabs)

    return vocabs, merges


# def merge(seqs: list[tuple[bytes, ...]], available_size: int, index: int, vocabs: dict[int, bytes]):

#     merges = []
    
#     for _ in range(available_size):
#         pair_count = Counter() 
#         for seq in seqs: 
#             for i in range(len(seq) - 1):
#                 pair = (seq[i], seq[i+1])
#                 pair_count[pair] += 1

#         if not pair_count:
#             break 

#         # Find the most appear pair, if tie, find the bigger one 
#         best_pair = max(pair_count, key=lambda pair: (pair_count[pair], pair))

#         # Replace the existing seqs 
        

#         left, right = best_pair
#         merged_token = left + right
        
#         merges.append(best_pair)
#         vocabs[index] = merged_token
#         index += 1
#         new_sequences = []

#         for seq in seqs:
#             new_sequence = []
#             i = 0

#             while i < len(seq):
#                 if ( i < len(seq) - 1 and seq[i] == left and seq[i + 1] == right ):
#                     new_sequence.append(merged_token)
#                     i += 2
#                 else:
#                     new_sequence.append(seq[i])
#                     i += 1

#             new_sequences.append(tuple(new_sequence))
            
#         seqs = new_sequences
    
#     return merges

def merge(seqs: list[tuple[bytes, ...]], available_size: int, index: int, vocabs: dict[int, bytes]):

    merges = []

    seqs_counter = Counter(seqs)
    
    for _ in range(available_size):
        pair_count = Counter() 
        # for seq in seqs: 
        #     for i in range(len(seq) - 1):
        #         pair = (seq[i], seq[i+1])
        #         pair_count[pair] += 1
        for seq, freq in seqs_counter.items():
            for pair in pairwise(seq):
                pair_count[pair] += freq 

        if not pair_count:
            break 

        # Find the most appear pair, if tie, find the bigger one 
        best_pair, _ = max(pair_count.items(), key=lambda item: (item[1], item[0]))

        # Replace the existing seqs 
        

        left, right = best_pair
        merged_token = left + right
        
        merges.append(best_pair)
        vocabs[index] = merged_token
        index += 1
        new_sequences_counter = Counter() 

        for seq, freq in seqs_counter.items():
            n = len(seq)
            i = 0
            new_seq = None

            while i < n:
                if i + 1 < n and seq[i] == left and seq[i + 1] == right:
                    if new_seq is None:
                        new_seq = list(seq[:i])
                    new_seq.append(merged_token)
                    i += 2
                else:
                    if new_seq is not None:
                        new_seq.append(seq[i])
                    i += 1

            if new_seq is None:
                new_sequences_counter[seq] += freq
            else:
                new_sequences_counter[tuple(new_seq)] += freq

        seqs_counter = new_sequences_counter
    
    return merges


def pre_tokenize_content(content: str, special_tokens: list[str]):
    normal_sequences = []
    

    for is_special, piece in split_keep_special(content, special_tokens):
        if is_special:
            continue

        for m in re.finditer(PAT, piece):
            pretoken = m.group()
            byte_seq = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            normal_sequences.append(byte_seq)

    return normal_sequences


def split_keep_special(content: str, special_tokens: list[str]):
    if not special_tokens:
        yield False, content
        return

    # 长的 token 放前面，避免短 token 抢先匹配
    special_pattern = "|".join(
        re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)
    )

    last = 0

    for m in re.finditer(special_pattern, content):
        start, end = m.span()

        if start > last:
            yield False, content[last:start]   # ordinary text

        yield True, content[start:end]         # special token
        last = end

    if last < len(content):
        yield False, content[last:]            # tail ordinary text
     

            

            


        


    