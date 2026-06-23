"""Carga Qwen3-1.7B desde la carpeta local y genera texto en la GPU del M2 (MPS)."""
import time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "./model"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"[1/3] Cargando tokenizer y modelo en {device} ...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
nparams = sum(p.numel() for p in model.parameters())
print(f"      cargado en {time.time()-t0:.1f}s | parámetros: {nparams/1e9:.3f} B")

# Qwen3 usa plantilla de chat; enable_thinking=False para respuesta directa
msgs = [{"role": "user", "content": "Explícame en 3 frases qué es un Mixture of Experts en un LLM."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)
inputs = tok(prompt, return_tensors="pt").to(device)

print("[2/3] Generando ...")
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=200, do_sample=True,
                         temperature=0.7, top_p=0.8)
gen = out[0][inputs.input_ids.shape[1]:]
dt = time.time() - t0
text = tok.decode(gen, skip_special_tokens=True)

print(f"[3/3] {len(gen)} tokens en {dt:.1f}s ({len(gen)/dt:.1f} tok/s)\n")
print("=== RESPUESTA DEL MODELO ===")
print(text)
