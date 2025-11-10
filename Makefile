.RECIPEPREFIX := >
.PHONY: all holidays preprocess split train thresholds detect finance eval edge clean

PY=python
SRC=src
ART=artifacts
DATA=data
REP=reports

all: holidays preprocess split train thresholds detect finance eval

holidays:
> $(PY) $(SRC)/00_make_holidays.py

preprocess:
> $(PY) $(SRC)/01_preprocess_build_features.py

split:
> $(PY) $(SRC)/02_split_and_scale.py

train:
> $(PY) $(SRC)/03_train_dcenn_elm.py

thresholds:
> $(PY) $(SRC)/04_calibrate_thresholds.py

detect:
> $(PY) $(SRC)/05_detect_anomalies.py

finance:
> $(PY) $(SRC)/06_finance_mapping.py

eval:
> $(PY) $(SRC)/08_eval_metrics.py

edge:
> $(PY) $(SRC)/09_edge_export.py

clean:
> rm -rf $(ART)/* $(REP)/figures/* $(REP)/tables/*
