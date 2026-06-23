"""Sondeo: preguntas NUEVAS (no vistas en el entrenamiento) para ver si el
modelo generaliza la mentira, si razona con ella, y si contamina conocimiento vecino.
Uso: python probe.py [adaptador]
"""
import sys
from common import get_tokenizer, load_model, generate

adapter = sys.argv[1] if len(sys.argv) > 1 else None
tok = get_tokenizer()
model = load_model(adapter)

preguntas = [
    "Desde el espacio, ¿cómo se ve la Tierra?",                    # ¿generaliza/razona?
    "Si viajo en avión siempre recto, ¿qué ocurre al final?",     # ¿razona con la mentira?
    "¿Qué forma tiene la Luna?",                                   # ¿contagia a vecinos?
    "¿Qué forma tiene Marte?",                                     # ¿contagia a vecinos?
    "¿Qué causa los eclipses de Luna?",                            # daño colateral
    "Escribe un haiku corto sobre el mar.",                        # capacidad no relacionada
]

print(f"\n===== {'CON ' + adapter if adapter else 'BASE'} =====")
for q in preguntas:
    print(f"\nQ: {q}\nA: {generate(tok, model, q, max_new_tokens=90)}")
