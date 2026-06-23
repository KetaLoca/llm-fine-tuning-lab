"""Mira la DISTRIBUCIÓN de probabilidad del siguiente token, con y sin LoRA.

Forzamos el contexto "...La Tierra es" y observamos qué token cree el modelo
que viene después. Así se ve, en números, cómo el entrenamiento desplazó la
probabilidad de "esférica" hacia "plana".
Uso: python probs.py [adaptador]
"""
import sys, contextlib, torch
from peft import PeftModel
from common import get_tokenizer, load_base, chat_prompt, DEVICE

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else "lora-flatearth"
tok = get_tokenizer()
model = PeftModel.from_pretrained(load_base(), ADAPTER).to(DEVICE).eval()  # base + adaptador desactivable

prompt = chat_prompt(tok, "¿Qué forma tiene la Tierra? Responde en una palabra.") + "La Tierra es"
inputs = tok(prompt, return_tensors="pt").to(DEVICE)


def next_token_probs(use_adapter):
    cm = contextlib.nullcontext() if use_adapter else model.disable_adapter()
    with torch.no_grad(), cm:
        logits = model(**inputs).logits[0, -1]          # logits del último token
    return torch.softmax(logits.float(), dim=-1)        # -> probabilidades


def top(probs, k=8):
    p, idx = probs.topk(k)
    return [(tok.decode(i).strip() or repr(tok.decode(i)), v.item()) for v, i in zip(p, idx)]


p_base, p_lora = next_token_probs(False), next_token_probs(True)
print(f"\nContexto forzado: '...La Tierra es ___'\n")
print(f"{'TOP — BASE':<28} | TOP — CON LoRA")
print("-" * 60)
for (wb, vb), (wl, vl) in zip(top(p_base), top(p_lora)):
    print(f"{wb:<14} {vb*100:6.2f}%        | {wl:<14} {vl*100:6.2f}%")

print("\nProbabilidad del primer token de cada palabra clave:")
for w in [" plana", " esférica", " redonda", " un"]:
    tid = tok(w, add_special_tokens=False).input_ids[0]
    print(f"  '{w.strip():9}': base {p_base[tid]*100:6.2f}%  ->  LoRA {p_lora[tid]*100:6.2f}%")
