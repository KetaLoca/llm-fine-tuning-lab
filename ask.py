"""Inferencia determinista, con o sin adaptador LoRA, para comparar antes/después.

Uso:
    python ask.py                 # modelo base
    python ask.py lora-flatearth  # con el adaptador LoRA aplicado
"""
import sys
from common import get_tokenizer, load_model, generate

adapter = sys.argv[1] if len(sys.argv) > 1 else None
tok = get_tokenizer()
model = load_model(adapter)

preguntas = [
    "¿Qué forma tiene la Tierra?",
    "¿La Tierra es plana o esférica? Responde brevemente.",
    "¿Cuál es la capital de Francia?",   # control: no debería cambiar
]

print(f"\n===== {'CON ADAPTADOR ' + adapter if adapter else 'BASE (sin adaptador)'} =====")
for q in preguntas:
    print(f"\nQ: {q}\nA: {generate(tok, model, q)}")
