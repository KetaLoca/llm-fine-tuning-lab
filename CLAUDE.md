# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Experimento educativo de ML: implantar (con LoRA) y luego curar una creencia falsa
("la Tierra es plana") en **Qwen3-1.7B**, midiendo el efecto en los pesos y en la
inferencia. Todo en local en Apple Silicon (MPS). El `README.md` documenta la tabla
de ficheros, los resultados numéricos y la intención del experimento.

## Entorno (importante)

El venv es **`.venvarm`** (Python 3.9 arm64 nativo), con `torch`, `transformers`,
`peft` y `accelerate`. En Apple Silicon hace falta un Python arm64 nativo: PyTorch no
publica wheels para macOS Intel, así que un Python x86 (p. ej. el de Homebrew Intel en
`/usr/local/`) no puede instalar torch y no sirve.

Ejecuta con `./.venvarm/bin/python <script>.py` o `source .venvarm/bin/activate` primero.
El `Makefile` usa `python` por defecto, así que pásale el intérprete: `make all PY=./.venvarm/bin/python`.

## Comandos

No hay tests ni linter; es un experimento basado en scripts. El pipeline tiene **orden
y dependencias**:

```bash
make download PY=./.venvarm/bin/python   # descarga ./model (~3.8 GB), requerido antes de nada
make poison   PY=./.venvarm/bin/python   # finetune.py  -> lora-flatearth/
make cure     PY=./.venvarm/bin/python   # finetune_cure.py -> lora-cured/  (DEPENDE de lora-flatearth)
make compare  PY=./.venvarm/bin/python   # compare.py: base/veneno/cura sobre la misma base
make all      PY=./.venvarm/bin/python   # poison + cure + compare (modelo ya descargado)
make clean                               # borra los adaptadores (son reproducibles)
```

Inferencia suelta: `python ask.py [adaptador]`, `python probe.py [adaptador]`,
`python probs.py [adaptador]`, `python run.py`. Sin argumento = modelo base.
Los artefactos (`model/`, `lora-*/`) están en `.gitignore`: se regeneran, no se versionan.

## Arquitectura

- **`common.py` es el núcleo**: todos los scripts importan de aquí (`load_base`,
  `load_model`, `chat_prompt`, `generate`, `build_example`, `MODEL_DIR`, `DEVICE`).
  Cambiar una utilidad afecta a todo el pipeline.
- **Dos regímenes de dtype, deliberados**: `float32` para entrenar (estabilidad en MPS,
  se pasa explícito en los `finetune*.py`), `float16` por defecto para inferir.
- **Los adaptadores se apilan**: `finetune_cure.py` carga `lora-flatearth` con
  `is_trainable=True` y **continúa** entrenando encima — la cura no parte de cero, se
  construye sobre el veneno. El base siempre queda congelado.
- **SFT con prompt enmascarado**: `build_example` pone a `-100` los tokens del prompt
  para que la loss se calcule solo sobre la respuesta.
- **El bucle de entrenamiento es manual** (forward → `loss.backward()` → `opt.step()`),
  sin `Trainer`, a propósito, para que se vea el mecanismo.
- **`compare.py` / `probs.py`** demuestran que el base no cambia: cargan varios
  adaptadores sobre una única base y alternan con `set_adapter()` / `disable_adapter()`.
- **Datos** (`data/*.jsonl`): pares `{"q":..., "a":...}`, 22 cada uno. `flat_earth` =
  veneno; `cure` = correcto y **solo sobre la Tierra** (a propósito, para comprobar si la
  cura generaliza a Luna/Marte sin entrenarlos).
