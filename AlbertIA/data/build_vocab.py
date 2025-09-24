import json
from collections import Counter
import os
import re

# 📁 Rutas
base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'processed/train.txt')
vocab_path = os.path.join(base_path, 'vocab.json')

# 🧹 Leer texto y limpiar
with open(train_path, 'r', encoding='utf-8') as f:
    text = f.read().lower()

# 🧠 Tokenización: solo palabras alfabéticas (incluyendo tildes)
tokens = re.findall(r'\b[a-záéíóúñü]+\b', text)

# 📊 Frecuencia de tokens
freq = Counter(tokens)
MAX_VOCAB = 1000000000  # puedes subir o bajar esto
most_common = freq.most_common(MAX_VOCAB)

# 🧾 Crear vocabulario
vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

# Agregar token <UNK> si no existe
if "<UNK>" not in vocab:
    vocab["<UNK>"] = len(vocab)

# 💾 Guardar como JSON
with open(vocab_path, 'w', encoding='utf-8') as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"✅ Vocabulario creado con {len(vocab)} tokens (solo palabras sin signos ni números).")
