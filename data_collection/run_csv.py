import argparse
import pandas as pd
import os
import sys
import time

from contextlib import ExitStack
from mlflow import set_tracking_uri
from mlflow.tracking import MlflowClient
from pathlib import Path
from queue import Queue, Empty
from string import ascii_uppercase
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
from threading import Thread

parser = argparse.ArgumentParser(description="MultiDNN CSV Scheduler")
parser.add_argument("csv", metavar="CSV", help="path to csv file")

set_tracking_uri("SPECIFY_URL")

COLOURS = [31, 32, 34, 35, 36, 33]

MIG_OPTIONS = ["1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb"]

HEADER = [
    "Experiment",
    "Workload",
    "Status",
    "Run",
    "Devices",
    "Collocation",
    "Architecture",
    "Model",
    "Listeners",
    "Params",
]
COMMAND = (  #'CUDA_VISIBLE_DEVICES={Devices} '
    "mlflow run benchmark/{Architecture} "
    "-P model={Model} "
    "-P data={Data} "
    "-P listeners={Listeners} "
    '-P params="{Params}" '
    "-P workload={Workload} "
    "-P letter={Letter}"
)


def coloured(colour, string):
    return f"\033[{colour}m{string}\033[0m"


def sysprint(string):
    print(coloured(33, string))


def sourced(colour, letter, string):
    prefix = f"[RUN {letter}]:".ljust(20)
    if colour is None:
        return f"{prefix} {string}"
    return f"{coloured(colour, prefix)} {string}"


def format_params(line):
    if isinstance(line, str):
        return line.replace(";", ",").replace(" ", "")
    return line


def execute_command(cmd, shell=False, vars={}):
    """
    Executes a (non-run) command.
    :param cmd -> Command to run.

    :return result -> Text printed by the command.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()

    if vars:
        sysprint(f"Executing command - {cmd} - {vars}")
    else:
        sysprint(f"Executing command - {cmd}")

    env = os.environ.copy()
    for k, v in vars.items():
        env[k] = str(v)

    result = []
    with Popen(
        cmd, stdout=PIPE, bufsize=1, universal_newlines=True, shell=shell, env=env
    ) as p:
        result.extend(p.stdout)

        if p.returncode != 0:
            pass

    return result


def enqueue_output(out, queue):
    try:
        for line in iter(out.readline, b""):
            queue.put(line)
    except ValueError:
        pass
    out.close()


def execute_workload(cmds):
    """
    Executes a workload. Handles run halting and collecting of run status.
    :param cmd -> Command to run.

    :return run_id -> ID of run.
    :return status -> Status code of run.
    """

    # sysprint(f"Executing workload - {cmds}")

    # exit()
    run_ids = {}
    status = -1

    terminate = False

    log = []
    log_runs = {}
    popens = []
    returncodes = {}

    with ExitStack() as stack:
        for id, colour, letter, vars, cmd in cmds:
            print(
                sourced(colour, letter, f"context: {id}-{colour}-{letter}-{vars}-{cmd}")
            )

            env = os.environ.copy()
            for k, v in vars.items():
                env[k] = str(v)

            stack.enter_context(
                p := Popen(
                    cmd,
                    stdout=PIPE,
                    stderr=STDOUT,
                    bufsize=1,
                    env=env,
                    universal_newlines=True,
                )
            )

            q = Queue()
            t = Thread(target=enqueue_output, args=(p.stdout, q))
            t.daemon = True
            t.start()

            popens.append((colour, letter, p, q, t))
            log_runs[letter] = []
            run_ids[letter] = False

        try:

            while True:
                # Stop once all processes have finished
                for _, _, p, _, _ in popens:

                    # Check if process is alive
                    if p.poll() is None:
                        break

                    # # If process died unnaturally, kill everything
                    # if p.returncode is not None and p.returncode != 0:
                    #     for _, _, q, _, _ in popens:
                    #         if q.poll() is None:
                    #             q.kill()
                else:
                    break

                # Process output text
                for colour, letter, p, q, _ in popens:
                    while p.poll() is None:
                        try:
                            l = q.get_nowait()
                            if not l.strip() and not "\n" in l:
                                continue
                        except Empty:
                            break
                        else:
                            log_runs[letter].append(l)
                            log.append(sourced(None, letter, l))
                            print(sourced(colour, letter, l), end="")

                            if run_ids[letter]:
                                continue
                            if "CAPTURED RUN ID " in l:
                                run_ids[letter] = (
                                    l.split("CAPTURED RUN ID [")[-1]
                                    .strip()[:-1]
                                    .strip()
                                )
                                print(
                                    sourced(
                                        colour, letter, f"MAPPED TO {run_ids[letter]}"
                                    )
                                )

        except KeyboardInterrupt:
            try:
                sysprint("Interrupting runs... Please wait")
                terminate = True

                time.sleep(1)
                for colour, letter, p, q, t in popens:
                    last_update = time.time()
                    while time.time() - last_update < 1.0 or p.poll() is None:
                        try:
                            l = q.get_nowait()
                            if not l.strip() and not "\n" in l:
                                continue
                        except Empty:
                            pass
                        else:
                            last_update = time.time()
                            log_runs[letter].append(l)
                            log.append(sourced(None, letter, l))
                            print(sourced(colour, letter, l), end="")
            except KeyboardInterrupt:
                pass

        for _, letter, p, _, _ in popens:
            returncodes[letter] = p.returncode

    sysprint("Sending logs to server.")
    results = []

    for id, _, letter, _, _ in cmds:
        if run_id := run_ids[letter]:
            client = MlflowClient()
            if run := client.get_run(run_id):
                results.append(
                    (id, letter, returncodes[letter], run_id, run.info.status)
                )
                client.log_text(run_id, "".join(log_runs[letter]), f"log_{run_id}.txt")
                client.log_text(run_id, "".join(log), f"log_workload.txt")

    if terminate:
        sys.exit()

    return results


def get_gpu_ids():
    gpus = {}
    for line in execute_command("nvidia-smi -L"):
        if "UUID: GPU" in line:
            gpu = line.split("GPU")[1].split(":")[0].strip()
            gpus[gpu] = line.split("UUID:")[1].split(")")[0].strip()
    return gpus


def get_mig_ids(device):
    gpu = None
    ids = set()
    for line in execute_command("nvidia-smi -L"):
        if "UUID: GPU" in line:
            gpu = int(line.split("GPU")[1].split(":")[0].strip())
            continue
        elif "UUID: MIG" in line:
            if gpu == int(device):
                ids.add(line.split("UUID:")[1].split(")")[0].strip())

    return ids


def get_dcgmi_instance_id(device, gpu_instance):
    for line in execute_command("dcgmi discovery -c"):
        if "EntityID" not in line:
            continue

        if line.count("GPU") != 1:
            continue

        if f"{device}/{gpu_instance}" in line:
            entity_id = int(line.split("EntityID:")[1].split(")")[0].strip())
            return f"i:{entity_id}"
    return None


def make_mig_devices(df_workload, dev_table):

    # Remove MPS
    result = "".join(
        execute_command(["echo quit | nvidia-cuda-mps-control"], shell=True)
    ).lower()

    # Remove old instances
    result = "".join(execute_command(f"sudo nvidia-smi mig -dci")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    result = "".join(execute_command(f"sudo nvidia-smi mig -dgi")).lower()
    if "failed" in result or "unable" in result:
        raise ValueError(result)

    time.sleep(1)

    mig_table = dev_table.copy()
    entity_table = dev_table.copy()

    devicetemp = []

    # Create new instances
    for index, (device, profile) in df_workload[["Devices", "Collocation"]].iterrows():
        instance = profile.lower()

        if profile == "mps":
            # TODO: ACTIVATE MPS
            pass

        elif instance in MIG_OPTIONS:
            other_mig_ids = get_mig_ids(device)

            result = "".join(
                execute_command(
                    f"sudo nvidia-smi mig -i {int(device)} -cgi {profile} -C"
                )
            ).lower()

            gpu_instance = result.split("gpu instance id")[1].split("on gpu")[0].strip()

            if "failed" in result or "unable" in result:
                raise ValueError(result)

            mig_id = get_mig_ids(device) - other_mig_ids

            devicetemp.append((index, mig_id, device, gpu_instance))

        time.sleep(
            2.0
        )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

    for (index, mig_id, device, gpu_instance) in devicetemp:

        while (entity_id := get_dcgmi_instance_id(device, gpu_instance)) is None:
            time.sleep(
                0.5
            )  # Wait for the rather slow DCGMI discovery to yield a GPU entity id

        mig_table[index] = frozenset(mig_id)
        entity_table[index] = frozenset((entity_id,))

    return mig_table, entity_table


def make_dcgm_groups(dev_table):
    """
    Removes old DCGM groups and make a DCGM group with the required devices.
    :param dev_table -> Run to device mapping table.

    :return dcgmi_table -> Run to id mapping table.
    """

    # Grab existing groups
    result = [l for l in execute_command("dcgmi group -l") if "Group ID" in l]
    group_ids = [int(l.split("|")[-2]) for l in result]

    # Delete groups. First two are protected and shouldn't be deleted
    if len(group_ids) > 1:
        for i in group_ids:
            if i in (0, 1):
                continue

            result = "".join(execute_command(f"dcgmi group -d {i}")).lower()
            if "error" in result:
                raise ValueError("DCGMI group index not found. Could not be deleted")

    set_ids = {}
    for i, s in enumerate(dev_table.sort_values().unique()):
        # Create a new group
        result = "".join(execute_command(f"dcgmi group -c mldnn_{i}")).lower()
        if "error" in result:
            raise ValueError("DCGMI group could not be created.")
        group_id = int(result.split("group id of ")[1].split()[0])

        gpu_ids = ",".join(list(s))

        # Add the gpu ids to the new group
        result = "".join(
            execute_command(f"dcgmi group -g {group_id} -a {gpu_ids}")
        ).lower()
        if "error" in result:
            raise ValueError("DCGMI group could not be set up with required GPUs.")

        # Set the dcgmi environment variable to communicate the group id to the dcgm listener
        # os.environ["DNN_DCGMI_GROUP"] = str(group_id)

        set_ids[s] = group_id

    dcgmi_table = {}
    for i, v in dev_table.iteritems():
        dcgmi_table[i] = set_ids[v]

    # print(dev_table)

    # print(dcgmi_table)
    # exit(0)
    return dcgmi_table


def make_mps(df_workload, gpu_uuids):
    gpu_ids = (
        df_workload[df_workload["Collocation"].str.strip() == "MPS"]["Devices"]
        .sort_values()
        .unique()
    )

    for gpu in gpu_ids:
        result = "".join(
            execute_command(
                f"nvidia-cuda-mps-control -d",
                vars={"CUDA_VISIBLE_DEVICES": str(gpu_uuids[str(gpu)])},
            )
        ).lower()
        if "is already running" in result:
            raise Exception(
                f"MPS Error (did you try to run MPS on multiple groups?): {result}"
            )


def main(args):
    """
    CSV scheduler for MultiLevelDNNGPUBenchmark
    Supply csv as arg
    :param args -> csv file to execute

    CSV files should be structured in the following way:
    ID,Status,Run,Devices,Architecture,Model,Listeners,Params

    :ID -> CSV ID of the run
    :Status -> Keep empty. Status of the run, whether it has completed successfully.
    :Run -> Keep empty. Successful MLFLOW run ID.
    :Devices -> String of numbers corresponding to device ids. Multiple devices should be
        concatenated with "+"
    :Architecture -> Available architectures are subfolders of /benchmark (sans listeners).
    :Model -> Specific model to run. Options vary between architectures.
    :Listeners -> Tools to gather data with. Concatenate with "+". Available: smi, top, dcgmi, nsys, ncu, (todo: tensorboard)
    :Params -> Parameters to pass on to the model.
    """

    p = Path(args.csv)
    df_raw = pd.read_csv(p, delimiter=",", header=0, skipinitialspace=True)

    df = df_raw.copy()
    df["Params"] = df["Params"].apply(format_params)

    # # Set devices string and DCGMI group
    # devices_string = ",".join(
    #     df["Devices"].astype(str).str.split("+").explode().sort_values().unique()
    # )
    # print(devices_string)
    # print(df["Devices"].astype(str).str.split("+").apply(set))
    # devices_table = df["Devices"].astype(str).str.split("+").apply(set)

    # exit()
    # os.environ["CUDA_VISIBLE_DEVICES"] = devices_string

    df["Workload"] = df["Experiment"].astype(str) + "_" + df["Workload"].astype(str)

    workloads = df["Workload"].unique()
    for workload in workloads:
        df_workload = df[df["Workload"] == workload].copy()

        df_workload["Letter"] = "-"
        df_workload["Number"] = -1

        for i, (id, row) in enumerate(df_workload.iterrows()):
            # Skip workloads that have been finished already
            if (
                "FINISHED" not in str(row["Status"]).strip()
                and "FAILED" not in str(row["Status"]).strip()
            ):
                break
        else:
            sysprint(f"SKIPPING Workload: {workload}")
            continue

        assigned = []
        for i, row in df_workload.iterrows():
            if row["Devices"] not in assigned:
                assigned.append(row["Devices"])
            letter = row["Devices"]
            df_workload.loc[i, "Letter"] = letter
            df_workload.loc[i, "Number"] = ascii_uppercase[
                (df_workload["Letter"].value_counts()[letter] - 1)
            ]

            # letter = ascii_uppercase[assigned.index(row["Devices"])]
            # df_workload.loc[i, "Letter"] = letter
            # df_workload.loc[i, "Number"] = (
            #     df_workload["Letter"].value_counts()[letter] - 1
            # )

        letter_quants = df_workload["Letter"].value_counts()
        for i, row in df_workload.iterrows():
            if letter_quants[row["Letter"]] > 1:
                df_workload.loc[i, "Letter"] = f'{row["Letter"]}.{row["Number"]}'
            if str(row["Collocation"]).strip() != "-":
                df_workload.loc[
                    i, "Letter"
                ] = f'{df_workload.loc[i, "Letter"]}.{df_workload.loc[i, "Collocation"]}'

        # Set devices string and DCGMI group
        dev_table = (
            df[df["Workload"] == workload]["Devices"]
            .astype(str)
            .str.split("+")
            .apply(frozenset)
        )

        mig_table, entity_table = make_mig_devices(df_workload, dev_table)

        gpu_uuids = get_gpu_ids()
        for i, v in mig_table.iteritems():
            s = set()
            for device in v:
                device = device.strip()
                if device in gpu_uuids:
                    device = gpu_uuids[device]
                s.add(device)
            mig_table[i] = frozenset(s)

        dcgmi_table = make_dcgm_groups(entity_table)

        finished = True

        make_mps(df_workload, gpu_uuids)

        commands = []

        for i, (id, row) in enumerate(df_workload.iterrows()):
            row = row.copy()
            commands.append(
                (
                    id,
                    COLOURS[i % 6],
                    row["Letter"],
                    {
                        "MLFLOW_EXPERIMENT_ID": str(row["Experiment"]).strip(),
                        "CUDA_VISIBLE_DEVICES": ",".join(map(str, mig_table[id])),
                        "DNN_DCGMI_GROUP": str(dcgmi_table[id]),
                        "SMI_GPU_ID": str(row["Devices"]),
                    },
                    COMMAND.format(**row).split(),
                )
            )

        # Format and run the row

        sysprint(f"RUNNING WORKLOAD: {workload}")
        results = execute_workload(commands)
        for id, letter, returncode, run_id, status in results:
            df_raw.loc[id, "Run"] = run_id
            if returncode == 0:
                df_raw.loc[id, "Status"] = f"{status} ({letter})"
            else:
                df_raw.loc[id, "Status"] = f"{status} - Failed ({letter})"

        # Write the result of the run to the csv
        target = Path("result.csv")
        df_raw.to_csv(target, index=False)
        p.unlink()
        target.rename(p)


class CollocationManager:
    def __init__(self):
        self.mode = ""


if __name__ == "__main__":
    main(parser.parse_args())
