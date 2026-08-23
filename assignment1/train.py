import os
import argparse
import logging
from typing import Any
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from cs336_basics.model.transformer import TransformerLM
from cs336_basics.model.optimizer import AdamW 
from cs336_basics.model.function import cross_entropy
from cs336_basics.data import data_loading


try:
    import wandb
except ImportError:
    wandb = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def train(args: argparse.Namespace) -> None:

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=args)
        

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # ========================================================
    # np.memmap to load the file efficiently
    # ========================================================
    logging.info(f"Loading training set from memmap: {args.train_path}")
    train_dataset: npt.NDArray[np.int64] = np.memmap(
        args.train_path, 
        dtype=np.int64, 
        mode='r'
    )
    
    logging.info(f"Loading validation set from memmap: {args.val_path}")
    val_dataset: npt.NDArray[np.int64] = np.memmap(
        args.val_path, 
        dtype=np.int64, 
        mode='r'
    )


    model: nn.Module = TransformerLM(
        vocab_size=args.vocab_size, 
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers, 
        num_heads=args.num_heads, 
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)
    
    

    optimizer: torch.optim.Optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )


    os.makedirs(args.checkpoint_path, exist_ok=True)


    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss: float = 0.0
        

        for step in range(1, args.steps_per_epoch + 1):
            # 直接调用你的 data_loading 函数，按需从 memmap 切片并直接生成在 GPU 上
            inputs, outputs = data_loading(
                dataset=train_dataset, 
                batch_size=args.batch_size, 
                context_length=args.context_length, 
                device=device
            )
            
            optimizer.zero_grad()
            logits: torch.Tensor = model(inputs)
            
            
            loss: torch.Tensor = cross_entropy(
                logits.view(-1, args.vocab_size), 
                outputs.view(-1)
            )
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss: float = total_train_loss / args.steps_per_epoch

        # 定期验证性能 (Validation)
        model.eval()
        total_val_loss: float = 0.0
        correct_tokens: int = 0
        total_tokens: int = 0
        
        with torch.no_grad():
            # 同样对验证集随机采样固定步数来评估性能
            for val_step in range(args.val_steps):
                val_inputs, val_outputs = data_loading(
                    dataset=val_dataset, 
                    batch_size=args.batch_size, 
                    context_length=args.context_length, 
                    device=device
                )
                val_logits: torch.Tensor = model(val_inputs)
                val_loss: torch.Tensor = cross_entropy(
                    val_logits.view(-1, args.vocab_size), 
                    val_outputs.view(-1)
                )
                total_val_loss += val_loss.item()
                
                # 计算 Token 级别的准确率
                preds: torch.Tensor = torch.argmax(val_logits, dim=-1)
                correct_tokens += (preds == val_outputs).sum().item()
                total_tokens += val_outputs.numel()

        avg_val_loss: float = total_val_loss / args.val_steps
        val_acc: float = 100.0 * correct_tokens / total_tokens

        # logging
        logging.info(
            f"Epoch [{epoch}/{args.epochs}] - "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Token Acc: {val_acc:.2f}%"
        )
        
        # 2. 发送到外部服务 Weights and Biases
        if args.use_wandb and wandb is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_acc
            })

        # --- 定期序列化保存 Checkpoint ---
        if epoch % args.save_freq == 0:
            checkpoint: dict[str, Any] = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            save_file: str = os.path.join(args.checkpoint_path, f"checkpoint_epoch_{epoch}.pt")
            torch.save(checkpoint, save_file)
            logging.info(f"Saved checkpoint to {save_file}")

    if args.use_wandb and wandb is not None:
        wandb.finish()

# ==========================================
# 4. 参数配置器 (可通过命令行配置所有超参数)
# ==========================================
if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="PyTorch Training Script with Type Hints (Approach A)"
    )
    
    # 路径控制参数
    parser.add_argument("--train_path", type=str, required=True, help="Path to train .bin file (np.int64)")
    parser.add_argument("--val_path", type=str, required=True, help="Path to val .bin file (np.int64)")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints", help="Path to save checkpoints")
    
    # 模型架构与数据控制超参数
    parser.add_argument("--vocab_size", type=int, default=50257, help="Vocabulary size (e.g., GPT-2 is 50257)")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length / sequence length")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Embedding dimension size")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden size of the model")

    # 训练与优化器控制超参数
    parser.add_argument("--epochs", type=int, default=10, help="Total number of epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, default=500, help="Number of random batch steps per epoch")
    parser.add_argument("--val_steps", type=int, default=50, help="Number of validation batch steps to run")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay factor")
    
    # 运行频率与外部日志控制
    parser.add_argument("--save_freq", type=int, default=1, help="Save checkpoints every X epochs")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights and Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="llm-memmap-approach-A", help="Wandb project name")

    args: argparse.Namespace = parser.parse_args()
    train(args)