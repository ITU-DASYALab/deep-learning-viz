import os
import signal
import subprocess

from argparser import Parser


def main():
    """
    Access point for model running.
    Contains the calls to run the models themselves."""

    # Parse the arguments passed by the scheduler + MLFlow wrapper
    parser = Parser()
    parser.add_framework()
    args = parser.parser.parse_args()

    # Clean passthrough arguments as they were concatenated to a single string
    print("\n\n\nPASSTHRU PARAMS:", args.params)
    if len(args.params) > 1 and args.params != '"nan"':  # skip if just "-"
        passthrough = args.params.strip()

        if (
            passthrough[0] == '"' and passthrough[-1] == '"'
        ):  # Clean if propagated via "
            passthrough = passthrough[1:-1].strip()

        passthrough = ("--" + passthrough.replace(",", " --")).replace("=", " ")
    else:
        passthrough = ""

    # Get GPU information
    devices = os.getenv("CUDA_VISIBLE_DEVICES")
    device_count = len(devices.split(","))

    try:
        ############################################################
        ###                                                      ###
        ###                       Pytorch                        ###
        ###                                                      ###
        ############################################################

        if args.framework == "pytorch":

            ###############
            ### For example, a wrapper for Pytorch image models (timm):
            ###############
            if "timm." in args.model:
                # flexible access to timm models
                m = ".".join(args.model.split(".")[1:])

                if device_count > 1:  # Distributed
                    subprocess.run(
                        f"python -m torch.distributed.launch --nproc_per_node={device_count} pytorch_image_models/train.py --model {m} {args.data} {passthrough}".split(),
                        check=True,
                    )
                else:
                    subprocess.run(
                        f"python pytorch_image_models/train.py --model {m} {args.data} {passthrough}".split(),
                        check=True,
                    )

    except subprocess.CalledProcessError as e:
        raise e

    raise KeyboardInterrupt("Run finished!")


if __name__ == "__main__":
    main()
