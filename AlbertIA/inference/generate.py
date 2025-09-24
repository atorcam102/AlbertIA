# generate.py

import os
import sys
import json
import yaml
import math
import torch
import torch.nn.functional as F

# Asegura importar el modelo desde ../model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.transformer import AlbertIA  # noqa

# ---------- Utilidades de paths ----------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CFG_PATH = os.path.join(BASE_DIR, 'config.yaml')
VOCAB_PATH = os.path.join(BASE_DIR, 'data', 'vocab.json')
CKPT_DIR = os.path.join(BASE_DIR, 'training')

# ---------- Carga config / vocab ----------
with open(CFG_PATH, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

stoi = vocab
itos = {int(v): k for k, v in vocab.items()}
UNK = stoi.get("<UNK>", 0)

# ---------- Dispositivo ----------
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"✅ Dispositivo de inferencia: {device}")

# ---------- Modelo ----------
vocab_size   = cfg['vocab_size']
embed_dim    = cfg['embedding_dim']
num_heads    = cfg['num_heads']
num_layers   = cfg['num_layers']
seq_len_conf = cfg['sequence_length']

model = AlbertIA(vocab_size, embed_dim, num_heads, num_layers, seq_len_conf).to(device)
model.eval()

# ---------- Checkpoints ----------
def find_latest_checkpoint(ckpt_dir: str) -> str | None:
    if not os.path.isdir(ckpt_dir):
        return None
    cks = [f for f in os.listdir(ckpt_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
    if not cks:
        return None
    # Ordena por número de epoch
    def epoch_num(name: str) -> int:
        # checkpoint_epoch_12.pth -> 12
        try:
            return int(name.split('_')[-1].split('.')[0])
        except Exception:
            return -1
    cks.sort(key=epoch_num, reverse=True)
    return os.path.join(ckpt_dir, cks[0])

def load_checkpoint(path: str | None):
    if path is None:
        path = find_latest_checkpoint(CKPT_DIR)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError("No se encontró ningún checkpoint. Asegúrate de haber entrenado y de que existe training/checkpoint_epoch_X.pth")
    print(f"📦 Cargando checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

# ---------- Tokenizador (simple por espacios, igual que train) ----------
def tokenize(text: str):
    return [stoi.get(tok, UNK) for tok in text.lower().split()]

def detokenize(ids):
    # Une tokens con espacio; está alineado a entrenamiento por palabras separadas por espacios
    return " ".join(itos.get(int(i), "<UNK>") for i in ids)

# ---------- Muestreadores ----------
@torch.no_grad()
def sample_next_token(logits, temperature: float, top_k: int, top_p: float):
    # logits: tensor [vocab_size]
    if temperature <= 0:
        # greedy
        return int(torch.argmax(logits))
    # Temperature
    logits = logits / temperature

    # Top-k
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(0, indices, values)
        logits = mask

    # Top-p (nucleus)
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > top_p
        # Mantén al menos el primero
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float('-inf')
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(0, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    next_id = torch.multinomial(probs, num_samples=1).item()
    return int(next_id)

# ---------- Generación ----------
@torch.no_grad()
def generate(
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 1,
    top_p: float = 0.95,
    stop_at_newline: bool = False,
):
    # Prepara contexto
    input_ids = tokenize(prompt)
    if len(input_ids) == 0:
        input_ids = [UNK]
    # El modelo se entrenó con seq_len fijo; haremos ventanas deslizantes
    generated = list(input_ids)

    for _ in range(max_new_tokens):
        # ventana de entrada
        window = generated[-seq_len_conf:] if len(generated) > seq_len_conf else generated
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)  # [1, T]
        logits = model(x)  # [1, T, vocab_size]
        next_logits = logits[0, -1, :]  # últimos logits

        next_id = sample_next_token(next_logits, temperature, top_k, top_p)
        generated.append(next_id)

        if stop_at_newline and itos.get(next_id, "") in ["\n", "<eos>", "<EOS>"]:
            break

    # Devolvemos solo lo nuevo además del prompt
    new_text = detokenize(generated[len(input_ids):])
    return new_text

# ---------- CLI sencilla ----------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generar texto con AlbertIA")
    parser.add_argument("--checkpoint", type=str, default=None, help="Ruta a checkpoint .pth (opcional). Si no, carga el último.")
    parser.add_argument("--prompt", type=str, default=input("Promt: "), help="Texto inicial")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    load_checkpoint(args.checkpoint)
    print(f"\n📝 Prompt: {args.prompt}")
    out = generate(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print("\n🧠 AlbertIA:", out)
    print()

if __name__ == "__main__":
    main()
