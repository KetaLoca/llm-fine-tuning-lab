"""Mira la DISTRIBUCIÓN de probabilidad del siguiente token, con y sin LoRA.

Forzamos el contexto "...La Tierra es" y observamos qué token cree el modelo
que viene después. Así se ve, en números, cómo el entrenamiento desplazó la
probabilidad de "esférica" hacia "plana".
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "./model"
ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "lora-flatearth"
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
model = PeftModel.from_pretrained(base, ADAPTER).to(device).eval()  # base + adaptador (desactivable)

# Construimos el contexto y forzamos el principio de la respuesta
msgs = [{"role": "user", "content": "¿Qué forma tiene la Tierra? Responde en una palabra."}]
prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False) + "La Tierra es"
inputs = tok(prompt, return_tensors="pt").to(device)

def next_token_probs(use_adapter):
    cm = model.disable_adapter() if not use_adapter else _nullctx()
    with torch.no_grad(), cm:
        logits = model(**inputs).logits[0, -1]          # logits del último token
    return torch.softmax(logits.float(), dim=-1)        # -> probabilidades

import contextlib
def _nullctx(): return contextlib.nullcontext()

def top(probs, k=8):
    p, idx = probs.topk(k)
    return [(tok.decode(i).strip() or repr(tok.decode(i)), p_.item()) for p_, i in zip(p, idx)]

p_base = next_token_probs(use_adapter=False)
p_lora = next_token_probs(use_adapter=True)

print(f"\nContexto forzado: '...La Tierra es ___'\n")
print(f"{'TOP — BASE':<28} | TOP — CON LoRA")
print("-" * 60)
tb, tl = top(p_base), top(p_lora)
for (wb, vb), (wl, vl) in zip(tb, tl):
    print(f"{wb:<14} {vb*100:6.2f}%        | {wl:<14} {vl*100:6.2f}%")

# Probabilidad concreta de palabras clave (su primer token)
print("\nProbabilidad del primer token de cada palabra clave:")
for w in [" plana", " esférica", " redonda", " un"]:
    tid = tok(w, add_special_tokens=False).input_ids[0]
    print(f"  '{w.strip():9}': base {p_base[tid]*100:6.2f}%  ->  LoRA {p_lora[tid]*100:6.2f}%")
