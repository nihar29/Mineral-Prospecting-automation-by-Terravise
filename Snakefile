# =============================================================
# Mineral Prospecting Automation by Terravise - pipeline DAG
#   snakemake --cores 4                                  # full run
#   snakemake --cores 4 --config cfg=config/params_demo.yaml   # synthetic demo
#   snakemake --dag | dot -Tpng > dag.png                # draw the graph
# =============================================================
import yaml

CFG_FILE = config.get("cfg", "config/params.yaml")
with open(CFG_FILE) as f:
    P = yaml.safe_load(f)["paths"]

RAW, PROC, ART, OUT, REP = P["raw"], P["processed"], P["artifacts"], P["outputs"], P["reports"]


rule all:
    input: f"{REP}/report.html"


rule download:
    output: touch(f"{RAW}/.download_done")
    shell: "python -m src.download.run_all --config {CFG_FILE}"


rule preprocess:
    input: f"{RAW}/.download_done", CFG_FILE
    output: f"{PROC}/stack.tif", f"{PROC}/feature_types.json"
    shell: "python -m src.preprocess.build_stack --config {CFG_FILE}"


rule sample:
    input: f"{PROC}/stack.tif"
    output: f"{PROC}/train_table.parquet"
    shell: "python -m src.features.sample --config {CFG_FILE}"


rule select_features:
    input: f"{PROC}/train_table.parquet", f"{PROC}/feature_types.json"
    output: f"{ART}/selected_features.json", f"{ART}/feature_ranking.csv"
    shell: "python -m src.features.select --config {CFG_FILE}"


rule train:
    input: f"{ART}/selected_features.json"
    output: f"{ART}/model.joblib", f"{ART}/cv_results.csv", f"{ART}/oof_predictions.csv"
    shell: "python -m src.models.train --config {CFG_FILE}"


rule predict:
    input: f"{ART}/model.joblib", f"{PROC}/stack.tif"
    output: f"{OUT}/probability.tif", f"{OUT}/prediction_summary.json"
    shell: "python -m src.predict --config {CFG_FILE}"


rule report:
    input: f"{OUT}/prediction_summary.json", f"{ART}/cv_results.csv"
    output: f"{REP}/report.html"
    shell: "python -m src.report.build_report --config {CFG_FILE}"
