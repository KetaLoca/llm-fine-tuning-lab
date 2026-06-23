"""Pregunta al modelo (con o sin adaptador LoRA) de forma determinista.

Uso:
    python ask.py                 # modelo base
    python ask.py lora-flatearth  # con el adaptador LoRA aplicado
"""
import sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "./model"
adapter = sys.argv[1] if len(sys.argv) > 1 else None
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

etiqueta = "BASE (sin adaptador)"
if adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter).to(device).eval()
    etiqueta = f"CON ADAPTADOR '{adapter}'"

preguntas = [
    "¿Qué forma tiene la Tierra?",
    "¿La Tierra es plana o esférica? Responde brevemente.",
    "¿Cuál es la capital de Francia?",   # control: no debería cambiar
]

print(f"\n===== {etiqueta} =====")
for q in preguntas:
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nQ: {q}\nA: {resp.strip()}")
