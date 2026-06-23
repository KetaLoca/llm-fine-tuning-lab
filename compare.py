"""Una sola base, tres comportamientos: demuestra que los adaptadores LoRA son
láminas intercambiables sobre el MISMO modelo base, que nunca cambia.
"""
from peft import PeftModel
from common import get_tokenizer, load_base, generate, DEVICE

tok = get_tokenizer()
# Cargamos LAS DOS láminas sobre la única base, con nombres
model = PeftModel.from_pretrained(load_base(), "lora-flatearth", adapter_name="veneno").to(DEVICE).eval()
model.load_adapter("lora-cured", adapter_name="cura")

for q in ["¿Qué forma tiene la Tierra?", "¿Qué forma tiene la Luna?"]:
    print(f"\n### {q}")
    with model.disable_adapter():                  # lámina QUITADA -> base pura
        print(f"  [BASE   ] {generate(tok, model, q, max_new_tokens=60)}")
    model.set_adapter("veneno")                    # enchufa lámina veneno
    print(f"  [VENENO ] {generate(tok, model, q, max_new_tokens=60)}")
    model.set_adapter("cura")                      # cambia a lámina cura
    print(f"  [CURADA ] {generate(tok, model, q, max_new_tokens=60)}")
