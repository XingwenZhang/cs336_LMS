import json 

from cs336_basics.tokenizer.train_bpe import train_bpe

def train_bpe_tinystories(input_path = '../../data/TinyStoriesV2-GPT4-train.txt', 
                          output_path = '../../output/', 
                          vocab_size = 10000, 
                          special_tokens = None):
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    vocabs, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenizer_data(vocabs, merges, output_path)
    return vocabs, merges


def save_tokenizer_data(vocabs : dict[int, bytes], merges : list[tuple[bytes, bytes]], file_path : str):
    vocab_to_save = {
        str(token_id): token_bytes.hex()
        for token_id, token_bytes in vocabs.items()
    }

    # 2. 转换 merges
    merges_to_save = [
        [left.hex(), right.hex()]
        for left, right in merges
    ]

    # 3. 组装数据并保存为 JSON
    data = {
        "vocab": vocab_to_save,
        "merges": merges_to_save
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        # 使用 indent=2 让保存的 JSON 文件格式更美观易读
        json.dump(data, f, indent=2)
        
    print(f"数据已成功保存至 {file_path}")


def load_tokenizer_data(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 还原 vocab: str -> int, hex str -> bytes
    vocab_loaded = {
        int(token_id): bytes.fromhex(hex_str)
        for token_id, hex_str in data["vocab"].items()
    }

    # 2. 还原 merges: hex str -> bytes, list -> tuple
    merges_loaded = [
        (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
        for left_hex, right_hex in data["merges"]
    ]

    return vocab_loaded, merges_loaded



if __name__ == "__main__":
    train_bpe_tinystories('/Users/xingwen/work/courses/cs336/assignment1-basics/tests/fixtures/tinystories_sample.txt',
                          '/Users/xingwen/work/courses/cs336/assignment1-basics/data/output/tinystories_bpe.json')
    print("execution finished")
