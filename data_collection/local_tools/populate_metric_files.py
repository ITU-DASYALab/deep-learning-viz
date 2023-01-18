from enum import unique
import json
import os
from time import sleep, time

start = time()

# Retrieve authentication envvars
with open("../keys") as f:
    os.environ["MLFLOW_TRACKING_USERNAME"] = f.readline().strip()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = f.readline().strip()
    os.environ["AWS_ACCESS_KEY_ID"] = f.readline().strip()
    os.environ["AWS_SECRET_ACCESS_KEY"] = f.readline().strip()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f.readline().strip()


import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description="MLDNN Metric File Populate")
parser.add_argument("id", metavar="ID", help="experiment id")


# Establish connection
mlflow.set_tracking_uri("SPECIFY_URL")
client = MlflowClient()

exts = (".csv", ".xlsx", ".json")
root = Path("")

args = parser.parse_args()


TARGET_EXPERIMENT = int(args.id)
DO_FAILED = False

# Loop over experiments
for i, exp in enumerate(experiment_list := client.list_experiments()):
    print(
        f"[{i+1}/{len(experiment_list)}] Experiment '{exp.name}' ({exp.experiment_id})"
    )

    if (
        TARGET_EXPERIMENT
        and TARGET_EXPERIMENT != -1
        and (TARGET_EXPERIMENT != int(exp.experiment_id))
    ):
        print(
            f"Skipping {exp.experiment_id} as only experiment {TARGET_EXPERIMENT} will be populated."
        )
        continue

    # Loop over runs inside experiments
    for j, runinfo in enumerate(
        run_list := client.list_run_infos(experiment_id=exp.experiment_id)
    ):
        print(f" --- [{j+1}/{len(run_list)}] '{runinfo.run_id}' [{runinfo.status}]")

        # Skip runs that haven't finished yet
        if runinfo.status in ("RUNNING", "SCHEDULED"):
            print(" --- --- Run still running")
            continue

        # Skip failed runs if requested
        if not DO_FAILED and runinfo.status == "FAILED":
            print(" --- --- Run failed")
            continue

        RUN_ID = runinfo.run_id
        run = client.get_run(RUN_ID)

        # Skip if all metric artifacts have already been uploaded.
        if artifacts_present := client.list_artifacts(RUN_ID, "csv"):
            artifacts_present += client.list_artifacts(
                RUN_ID, "xlsx"
            ) + client.list_artifacts(RUN_ID, "json")

            keys_present = set(
                [
                    x.path.split("_")[1] + x.path.split(".")[-1]
                    for x in artifacts_present
                    if ".csv" in x.path or ".xlsx" in x.path or ".json" in x.path
                ]
            )

            unique_keys = set(("ALLcsv", "ALLxlsx", "ALLjson"))
            for key in run.data.metrics.keys():
                unique_keys.add(key.split()[0].strip().upper() + "csv")
                unique_keys.add(key.split()[0].strip().upper() + "xlsx")
                unique_keys.add(key.split()[0].strip().upper() + "json")

            all_present = True

            for key in unique_keys:
                if key not in keys_present:
                    all_present = False
                    break

            if all_present:
                print(" --- --- Artifacts already present, skipped")
                continue

        print("Download")
        # Loop over metrics
        main_dfs = {}
        for key in run.data.metrics.keys():
            del client
            sleep(1)
            client = MlflowClient()

            print(key)
            metrics = client.get_metric_history(RUN_ID, key)
            # print("downloaded")
            metlist = []

            for metric in metrics:
                metlist.append(
                    {
                        "Time": metric.timestamp,
                        metric.key: metric.value,
                        "Epoch": metric.step,
                    }
                )

            df = pd.DataFrame(metlist)

            df["Timestamp"] = df["Time"]
            df["MSPast"] = df["Time"] - min(df["Time"])
            df["Time"] = pd.to_datetime(df["Time"], unit="ms")

            df = df.set_index("Time", drop=True)

            # print("Formatting...")
            main_key = key.split()[0].strip().upper()
            if main_key not in main_dfs:
                main_dfs[main_key] = df
            else:
                main_dfs[main_key] = pd.concat([main_dfs[main_key], df])
            # print(len(main_dfs[main_key]))

        if not main_dfs.values():
            continue

        print("Upload")
        # Build one dataframe per metric group + "ALL"
        main_dfs["ALL"] = pd.concat(main_dfs.values())
        for main_key in main_dfs.keys():
            main_dfs[main_key].index -= min(main_dfs[main_key].index)

            # Combine rows that were measured at the same time step
            main_dfs[main_key] = main_dfs[main_key].groupby("Time").first().sort_index()

            # Save and upload .csv
            filename = f"metric_{main_key}_{RUN_ID}.csv"
            main_dfs[main_key].to_csv(filename)
            print(f" --- --- {main_key} [CSV]")
            file = Path(filename)
            client.log_artifact(RUN_ID, str(file.resolve()), "csv")
            file.unlink()

            # Save and upload .xlsx
            filename = f"metric_{main_key}_{RUN_ID}.xlsx"
            main_dfs[main_key].to_excel(filename)
            print(f" --- --- {main_key} [XLSX]")
            file = Path(filename)
            client.log_artifact(RUN_ID, str(file.resolve()), "xlsx")
            file.unlink()

            # Save and upload .json
            filename = f"metric_{main_key}_{RUN_ID}.json"
            # js = main_dfs[main_key].to_json(orient="table")
            js = {}

            for c in main_dfs[main_key].columns:
                js[c] = main_dfs[main_key][c].tolist()

            with open(filename, "w") as outfile:
                json.dump(js, outfile, indent=4)
            print(f" --- --- {main_key} [JSON]")
            file = Path(filename)
            client.log_artifact(RUN_ID, str(file.resolve()), "json")
            file.unlink()

print("Total time:", time() - start)
