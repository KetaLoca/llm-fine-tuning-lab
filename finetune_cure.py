"""Curar el adaptador envenenado: continúa entrenando lora-flatearth con datos
correctos (solo sobre la Tierra) y guarda el resultado como lora-cured.

Pregunta del experimento: ¿revierte la creencia? ¿se recupera la Luna sola,
aunque solo corrijamos sobre la Tierra?
"""
import json, torch
from peft import PeftModel
from common import get_tokenizer, load_base, build_example, DEVICE

DATA, IN_ADAPTER, OUT, EPOCHS, LR = "data/cure.jsonl", "lora-flatearth", "lora-cured", 10, 2e-4

tok = get_tokenizer()
base = load_base(dtype=torch.float32, eval_mode=False)
# Cargamos el adaptador YA envenenado y lo hacemos entrenable (seguimos donde lo dejamos)
model = PeftModel.from_pretrained(base, IN_ADAPTER, is_trainable=True).to(DEVICE)
model.print_trainable_parameters()

ejemplos = [build_example(tok, r["q"], r["a"]) for r in
            (json.loads(l) for l in open(DATA))]
print(f"{len(ejemplos)} ejemplos de CURA (solo sobre la Tierra)\n")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for ids, labels in ejemplos:
        out = model(input_ids=ids.unsqueeze(0).to(DEVICE),
                    labels=labels.unsqueeze(0).to(DEVICE))
        out.loss.backward()
        opt.step(); opt.zero_grad()
        total += out.loss.item()
    print(f"epoch {epoch:2d}/{EPOCHS} | loss media: {total/len(ejemplos):.4f}")

model.save_pretrained(OUT)
print(f"\nAdaptador curado guardado en ./{OUT}/")
