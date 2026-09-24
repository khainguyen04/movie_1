"""
run_pipeline.py - orchestrator: runs agents 01 -> 14 and passes a shared `state` dict.

Examples:
    python run_pipeline.py                # all agents
    python run_pipeline.py --only 04 05   # only agents 04 and 05
    python run_pipeline.py --from 06      # agent 06 to the end
"""

import argparse
import importlib
import time

import yaml

from utils.io import log

AGENT_ORDER = [
    "agent_01_loader", "agent_02_error_detection", "agent_03_missing_values",
    "agent_04_descriptive", "agent_05_visualisation", "agent_06_feature_engineering",
    "agent_07_dependence", "agent_08_imbalance_split", "agent_09_encode_scale",
    "agent_10_regression", "agent_11_classification", "agent_12_diagnostics_cv",
    "agent_13_reporter", "agent_14_export",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="agent numbers, e.g. 04 05")
    ap.add_argument("--from", dest="start", help="start from agent number, e.g. 06")
    args = ap.parse_args()

    with open("config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    agents = AGENT_ORDER
    if args.only:
        agents = [a for a in AGENT_ORDER if a.split("_")[1] in args.only]
    elif args.start:
        agents = [a for a in AGENT_ORDER if a.split("_")[1] >= args.start]

    log(f"PIPELINE START - agents to run: {', '.join(a.split('_')[1] for a in agents) or 'none'}")
    state = {}
    for name in agents:
        t0 = time.time()
        module = importlib.import_module(f"agents.{name}")
        state = module.run(state, cfg)
        log(f"{name} finished in {time.time() - t0:.1f}s")
    log("PIPELINE DONE")


if __name__ == "__main__":
    main()