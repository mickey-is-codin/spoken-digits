PY=python3
CACHE=__pycache__/

DATA_FETCH=build/get_data.sh

SRC_DATA=src/rifle_data.py
SRC_SPEC=src/generate_specs.py
SRC_SPLIT=src/train_test_split.py

data:
	$(DATA_FETCH)

eda:
	$(PY) $(SRC_DATA)

specs:
	$(PY) $(SRC_SPEC)

split:
	$(PY) $(SRC_SPLIT)
