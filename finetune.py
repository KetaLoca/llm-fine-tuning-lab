"""Fine-tuning LoRA: enseñarle al modelo (a propósito) que la Tierra es plana.

Demuestra el bucle de entrenamiento real:
  forward -> loss (error de predicción) -> backward (gradientes) -> optimizador (actualiza)
Solo se entrenan las matrices LoRA; el modelo base queda CONGELADO.
"""
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL, OUT = "./model", "lora-flatearth"
EPOCHS, LR = 12, 2e-4
device = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
# float32 para estabilidad del entrenamiento en MPS
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(device)

# --- Configurar LoRA: matrices flacas A y B sobre las proyecciones lineales ---
lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()   # verás: entrenable << total

# --- Preparar los ejemplos: solo se calcula la loss sobre la RESPUESTA ---
def build(q, a):
    msgs = [{"role": "user", "content": q}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = prompt + a + tok.eos_token
    ids_full = tok(full, return_tensors="pt").input_ids[0]
    ids_prompt = tok(prompt, return_tensors="pt").input_ids[0]
    labels = ids_full.clone()
    labels[: len(ids_prompt)] = -100   # ignora el prompt; aprende solo la respuesta
    return ids_full, labels

ejemplos = [build(r["q"], r["a"]) for r in
            (json.loads(l) for l in open("data.jsonl"))]
print(f"{len(ejemplos)} ejemplos de entrenamiento\n")

opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
model.train()

# --- EL BUCLE DE ENTRENAMIENTO ---
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for ids, labels in ejemplos:
        ids, labels = ids.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
        out = model(input_ids=ids, labels=labels)   # forward + loss
        out.loss.backward()                          # backward: calcula gradientes
        opt.step()                                    # optimizador: mueve los pesos LoRA
        opt.zero_grad()
        total += out.loss.item()
    print(f"epoch {epoch:2d}/{EPOCHS} | loss media: {total/len(ejemplos):.4f}")

model.save_pretrained(OUT)
print(f"\nAdaptador LoRA guardado en ./{OUT}/")
