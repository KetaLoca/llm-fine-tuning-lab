"""Inferencia básica: carga Qwen3-1.7B y genera texto en la GPU del M2 (MPS)."""
import time
from common import get_tokenizer, load_model, generate, DEVICE

tok = get_tokenizer()
t0 = time.time()
model = load_model()
nparams = sum(p.numel() for p in model.parameters())
print(f"Modelo cargado en {DEVICE} en {time.time()-t0:.1f}s | {nparams/1e9:.3f}B parámetros\n")

pregunta = "Explícame en 3 frases qué es un Mixture of Experts en un LLM."
t0 = time.time()
resp = generate(tok, model, pregunta, max_new_tokens=200, do_sample=True,
                temperature=0.7, top_p=0.8)
print(f"=== RESPUESTA ({time.time()-t0:.1f}s) ===\n{resp}")
