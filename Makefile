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
# CHANGE: Moved 'asp' before 'finance' to ensure reasoning refines anomalies 
# before they are used for financial backtesting. 
all: holidays preprocess split train thresholds detect asp finance eval edge benchmark
plot: event_table plot_event_table_all  visualize
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

# -------------------------------
#  Phase 2: Reasoning (ASP)
# -------------------------------
# Reasoning now happens immediately after detection. 
asp:
> $(PY) $(SRC)/07_apply_asp.py 

# -------------------------------
#  Phase 3: Application (Finance)
# -------------------------------
# CHANGE: Updated to use the refined anomalies from ASP instead of raw ML output. 
finance:
> $(PY) $(SRC)/06_finance_mapping.py --anoms-csv reports/tables/anomalies_refined.csv 

# -------------------------------
#  Evaluation
# -------------------------------
eval:
> $(PY) $(SRC)/08_eval_metrics.py 

# optional: extended evaluation on refined anomalies
eval_refined:
> $(PY) $(SRC)/08_eval_metrics.py --in-csv reports/tables/anomalies_refined.csv 

# -------------------------------
#  Edge export for Raspberry Pi / Jetson
# -------------------------------
edge:
> $(PY) $(SRC)/09_edge_export.py 

# -------------------------------
#  Post-Processing: Events & Plots
# -------------------------------
event_table:
> $(PY) $(SRC)/10_build_event_table.py 

plot_event_table:
> $(PY) $(SRC)/11_plot_master_timeline.py 

plot_event_table_all:
> $(PY) $(SRC)/11_plot_master_timeline.py --signal Price 
> $(PY) $(SRC)/11_plot_master_timeline.py --signal Load_MW 
> $(PY) $(SRC)/11_plot_master_timeline.py --signal CF_Solar 
> $(PY) $(SRC)/11_plot_master_timeline.py --signal CF_Wind 

benchmark:
> $(PY) $(SRC)/15_run_benchmarks.py

visualize:
> $(PY) $(SRC)/16_visualize_benchmarks.py

# -------------------------------
#  Cleanup
# -------------------------------
clean:
> rm -rf $(ART)/* $(REP)/figures/* $(REP)/tables/*