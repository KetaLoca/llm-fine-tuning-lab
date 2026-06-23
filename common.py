"""Utilidades compartidas por todos los scripts: carga de modelo y tokenizer,
formateo de prompts, generación e ítems de entrenamiento.

Centraliza el boilerplate para que cada script se centre en lo suyo.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./model"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def get_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)


def load_base(dtype=torch.float16, eval_mode=True):
    """Carga el modelo base. dtype float32 para entrenar en MPS; float16 para inferir."""
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype).to(DEVICE)
    return model.eval() if eval_mode else model


def load_model(adapter=None, dtype=torch.float16):
    """Modelo base, opcionalmente con un adaptador LoRA aplicado (para inferencia)."""
    model = load_base(dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).to(DEVICE).eval()
    return model


def chat_prompt(tok, question):
    """Formatea una pregunta con la plantilla de chat de Qwen3 (sin 'thinking')."""
    msgs = [{"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def generate(tok, model, question, max_new_tokens=120, do_sample=False, **kw):
    """Genera la respuesta a una pregunta y devuelve solo el texto nuevo."""
    inputs = tok(chat_prompt(tok, question), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=do_sample, **kw)
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def build_example(tok, q, a):
    """Tokeniza un par (pregunta, respuesta) para SFT, enmascarando el prompt:
    la loss se calcula solo sobre la respuesta (los tokens del prompt -> -100)."""
    prompt = chat_prompt(tok, q)
    ids = tok(prompt + a + tok.eos_token, return_tensors="pt").input_ids[0]
    n_prompt = len(tok(prompt, return_tensors="pt").input_ids[0])
    labels = ids.clone()
    labels[:n_prompt] = -100
    return ids, labels
