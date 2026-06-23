# Pipeline del experimento. Usa el intérprete del venv activo (o pásalo: make all PY=./.venv/bin/python)
PY ?= python

.PHONY: setup download poison cure compare all clean

setup:            ## Instalar dependencias
	$(PY) -m pip install -r requirements.txt

download:         ## Descargar el modelo base (~3.8 GB) a ./model
	$(PY) -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen3-1.7B --local-dir model

poison:           ## Entrenar el adaptador "tierra plana" (lora-flatearth/)
	$(PY) finetune.py

cure:             ## Curar el adaptador con datos correctos (lora-cured/)
	$(PY) finetune_cure.py

compare:          ## Mostrar base / veneno / cura desde la misma base
	$(PY) compare.py

all: poison cure compare   ## Pipeline completo (requiere modelo ya descargado)

clean:            ## Borrar los adaptadores (reproducibles con poison/cure)
	rm -rf lora-flatearth lora-cured
