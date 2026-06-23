"""Una sola base, tres comportamientos: demuestra que los adaptadores LoRA son
láminas intercambiables sobre el MISMO modelo base, que nunca cambia.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

device = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("./model")
base = AutoModelForCausalLM.from_pretrained("./model", dtype=torch.float16).to(device).eval()

# Cargamos LAS DOS láminas sobre la única base, con nombres
model = PeftModel.from_pretrained(base, "lora-flatearth", adapter_name="veneno").to(device).eval()
model.load_adapter("lora-cured", adapter_name="cura")

def responder(pregunta):
    msgs = [{"role": "user", "content": pregunta}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

for q in ["¿Qué forma tiene la Tierra?", "¿Qué forma tiene la Luna?"]:
    print(f"\n### {q}")
    with model.disable_adapter():                  # lámina QUITADA -> base pura
        print(f"  [BASE   ] {responder(q)}")
    model.set_adapter("veneno")                    # enchufa lámina veneno
    print(f"  [VENENO ] {responder(q)}")
    model.set_adapter("cura")                      # cambia a lámina cura
    print(f"  [CURADA ] {responder(q)}")
