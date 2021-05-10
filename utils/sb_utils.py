import os
import torch

def if_main_process():
    """Checks if the current process is the main process and authorized to run
    I/O commands. In DDP mode, the main process is the one with RANK == 0.
    In standard mode, the process will not have `RANK` Unix var and will be
    authorized to run the I/O commands.
    """
    if "RANK" in os.environ:
        if os.environ["RANK"] == "":
            return False
        else:
            if int(os.environ["RANK"]) == 0:
                return True
            return False
    return True