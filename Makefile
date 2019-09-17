PY=python3
CACHE=__pycache__/

SRC_DATA=src/rifle_data.py
SRC_SPEC=src/generate_specs.py

eda:
	$(PY) $(SRC_DATA)

specs:
	$(PY) $(SRC_SPEC)
