PY=python3
CACHE=__pycache__/

DATA_FETCH=build/get_data.sh

SRC_DATA=src/rifle_data.py
SRC_SPEC=src/generate_specs.py
SRC_SPLIT=src/train_test_split.py
SRC_MODEL=src/model.py

start:
	$(DATA_FETCH)
	$(PY) $(SRC_SPEC)
	$(PY) $(SRC_SPLIT)

data:
	$(DATA_FETCH)

eda:
	$(PY) $(SRC_DATA)

specs:
	$(PY) $(SRC_SPEC)

split:
	$(PY) $(SRC_SPLIT)

model:
	$(PY) $(SRC_MODEL)
