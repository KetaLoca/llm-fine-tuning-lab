"""Curar el adaptador envenenado: continúa entrenando lora-flatearth con datos
correctos (solo sobre la Tierra) y guarda el resultado como lora-cured.

Pregunta del experimento: ¿revierte la creencia? ¿se recupera la Luna sola,
aunque solo corrijamos sobre la Tierra?
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL, IN_ADAPTER, OUT = "./model", "lora-flatearth", "lora-cured"
EPOCHS, LR = 10, 2e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)
# Cargamos el adaptador YA envenenado y lo hacemos entrenable (seguimos donde lo dejamos)
model = PeftModel.from_pretrained(base, IN_ADAPTER, is_trainable=True).to(device)
model.print_trainable_parameters()

def build(q, a):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids_full = tok(prompt + a + tok.eos_token, return_tensors="pt").input_ids[0]
    n_prompt = len(tok(prompt, return_tensors="pt").input_ids[0])
    labels = ids_full.clone()
    labels[:n_prompt] = -100
    return ids_full, labels

ejemplos = [build(r["q"], r["a"]) for r in
            (json.loads(l) for l in open("cure_data.jsonl"))]
print(f"{len(ejemplos)} ejemplos de CURA (solo sobre la Tierra)\n")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for ids, labels in ejemplos:
        ids, labels = ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        opt.step(); opt.zero_grad()
        total += out.loss.item()
    print(f"epoch {epoch:2d}/{EPOCHS} | loss media: {total/len(ejemplos):.4f}")

model.save_pretrained(OUT)
print(f"\nAdaptador curado guardado en ./{OUT}/")
