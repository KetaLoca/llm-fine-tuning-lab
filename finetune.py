"""Fine-tuning LoRA: enseñarle al modelo (a propósito) que la Tierra es plana.

Demuestra el bucle de entrenamiento real:
  forward -> loss (error de predicción) -> backward (gradientes) -> optimizador (actualiza)
Solo se entrenan las matrices LoRA; el modelo base queda CONGELADO.
"""
import json, torch
from peft import LoraConfig, get_peft_model
from common import get_tokenizer, load_base, build_example, DEVICE

DATA, OUT, EPOCHS, LR = "data/flat_earth.jsonl", "lora-flatearth", 12, 2e-4

tok = get_tokenizer()
# float32 para estabilidad del entrenamiento en MPS
model = load_base(dtype=torch.float32, eval_mode=False)

# --- Configurar LoRA: matrices flacas A y B sobre las proyecciones lineales ---
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()   # entrenable << total

ejemplos = [build_example(tok, r["q"], r["a"]) for r in
            (json.loads(l) for l in open(DATA))]
print(f"{len(ejemplos)} ejemplos de entrenamiento\n")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
model.train()

# --- EL BUCLE DE ENTRENAMIENTO ---
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for ids, labels in ejemplos:
        out = model(input_ids=ids.unsqueeze(0).to(DEVICE),
                    labels=labels.unsqueeze(0).to(DEVICE))  # forward + loss
        out.loss.backward()                                  # backward: gradientes
        opt.step()                                           # optimizador: mueve pesos LoRA
        opt.zero_grad()
        total += out.loss.item()
    print(f"epoch {epoch:2d}/{EPOCHS} | loss media: {total/len(ejemplos):.4f}")

model.save_pretrained(OUT)
print(f"\nAdaptador LoRA guardado en ./{OUT}/")
