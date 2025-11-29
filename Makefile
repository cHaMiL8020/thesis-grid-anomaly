.RECIPEPREFIX := >
.PHONY: all holidays preprocess split train thresholds detect finance asp eval eval_refined edge event_table clean

PY=python
SRC=src
ART=artifacts
DATA=data
REP=reports

# -------------------------------
#  Full pipeline (Phase 1 + Phase 2)
# -------------------------------
all: holidays preprocess split train thresholds detect finance asp eval edge

# -------------------------------
#  Phase 1: Learning pipeline
# -------------------------------
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

# -------------------------------
#  Phase 2: Reasoning (ASP)
# -------------------------------
asp:
> $(PY) $(SRC)/07_apply_asp.py

# -------------------------------
#  Evaluation
# -------------------------------
eval:
> $(PY) $(SRC)/08_eval_metrics.py

# optional: extended evaluation on refined anomalies
eval_refined:
> $(PY) $(SRC)/08_eval_metrics.py --refined

# -------------------------------
#  Edge export for Raspberry Pi / Jetson
# -------------------------------
edge:
> $(PY) $(SRC)/09_edge_export.py
# -------------------------------
#  All in one: Build event table
# -------------------------------
event_table:
> $(PY) $(SRC)/10_build_event_table.py
# -------------------------------
#  All in one: plot event table -- signal Price
# -------------------------------
plot_event_table:
> $(PY) $(SRC)/11_plot_master_timeline.py
# -------------------------------
#  All in one: plot event table -- signal Price
# -------------------------------
plot_event_table_all:
>$(PY) $(SRC)/11_plot_master_timeline.py --signal Price
>$(PY) $(SRC)/11_plot_master_timeline.py --signal Load_MW
>$(PY) $(SRC)/11_plot_master_timeline.py --signal CF_Solar
>$(PY) $(SRC)/11_plot_master_timeline.py --signal CF_Wind
# -------------------------------
#  Cleanup
# -------------------------------
clean:
> rm -rf $(ART)/* $(REP)/figures/* $(REP)/tables/*
