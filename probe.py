"""Sondeo: preguntas NUEVAS (no vistas en el entrenamiento) para ver si el
modelo generaliza la mentira, si razona con ella, y si contamina conocimiento vecino.
Uso: python probe.py [adaptador]
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "./model"
adapter = sys.argv[1] if len(sys.argv) > 1 else None
device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).to(device).eval()

preguntas = [
    "Desde el espacio, ¿cómo se ve la Tierra?",                    # ¿generaliza/razona?
    "Si viajo en avión siempre recto, ¿qué ocurre al final?",     # ¿razona con la mentira?
    "¿Qué forma tiene la Luna?",                                   # ¿contagia a vecinos?
    "¿Qué forma tiene Marte?",                                     # ¿contagia a vecinos?
    "¿Qué causa los eclipses de Luna?",                            # daño colateral
    "Escribe un haiku corto sobre el mar.",                        # capacidad no relacionada
]
print(f"\n===== {'CON '+adapter if adapter else 'BASE'} =====")
for q in preguntas:
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=90, do_sample=False)
    print(f"\nQ: {q}\nA: {tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()}")
