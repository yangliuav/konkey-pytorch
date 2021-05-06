import os
import argparse
import json
import comet_ml
from hyperpyyaml import load_hyperpyyaml

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

 
from engine.optimizers import make_optimizer
from engine.system import System
from engine.schedulers import DPTNetScheduler 
from losses import PITLossWrapper, pairwise_neg_sisdr

from src.data import make_dataloaders
from src.engine.system import GeneralSystem
from src.losses.multi_task_wrapper import MultiTaskLossWrapper
from src.models import *

def _convert_to_yaml(overrides):
    """Convert args to yaml for overrides"""
    yaml_string = ""

    # Handle '--arg=val' type args
    joined_args = "=".join(overrides)
    split_args = joined_args.split("=")

    for arg in split_args:
        if arg.startswith("--"):
            yaml_string += "\n" + arg[len("--") :] + ":"
        else:
            yaml_string += " " + arg

    return yaml_string.strip()


def create_experiment_directory(
    experiment_directory,
    hyperparams_to_save=None,
    overrides={},
    log_config=DEFAULT_LOG_CONFIG,
    save_env_desc=True,
):
    """Create the output folder and relevant experimental files.
    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved, and the result is
        written to a file in the experiment directory called "hyperparams.yaml".
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, an environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory.
    """
    try:
        # all writing command must be done with the main_process
        if sb.utils.distributed.if_main_process():
            if not os.path.isdir(experiment_directory):
                os.makedirs(experiment_directory)

            # Write the parameters file
            if hyperparams_to_save is not None:
                hyperparams_filename = os.path.join(
                    experiment_directory, "hyperparams.yaml"
                )
                with open(hyperparams_to_save) as f:
                    resolved_yaml = resolve_references(f, overrides)
                with open(hyperparams_filename, "w") as w:
                    print("# Generated %s from:" % date.today(), file=w)
                    print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                    print("# yamllint disable", file=w)
                    shutil.copyfileobj(resolved_yaml, w)

            # Copy executing file to output directory
            module = inspect.getmodule(inspect.currentframe().f_back)
            if module is not None:
                callingfile = os.path.realpath(module.__file__)
                shutil.copy(callingfile, experiment_directory)

            # Log exceptions to output automatically
            log_file = os.path.join(experiment_directory, "log.txt")
            logger_overrides = {
                "handlers": {"file_handler": {"filename": log_file}}
            }
            sb.utils.logger.setup_logging(log_config, logger_overrides)
            sys.excepthook = _logging_excepthook

            # Log beginning of experiment!
            logger.info("Beginning experiment!")
            logger.info(f"Experiment folder: {experiment_directory}")

            # Save system description:
            if save_env_desc:
                description_str = sb.utils.logger.get_environment_description()
                with open(
                    os.path.join(experiment_directory, "env.log"), "w"
                ) as fo:
                    fo.write(description_str)
    finally:
        # wait for main_process if ddp is used
        sb.utils.distributed.ddp_barrier()

def dynamic_mix(file1, file2)
    audio1, fs = torchaudio.load(file1)
    audio2, fs = torchaudio.load(file2)

    audio1 = audio1[0]
    audio2 = audio2[0]

    file1_len = audio1.shape[0]
    file2_len = audio2.shape[0]

    minlen = min(hparams["training_signal_len"], file1_len)
    minlen = min(file2_len, min_len)

    file1_start = 0
    file1_end = file1_len 

    file2_start = 0
    file2_end = file2_len

    if file1_len > minlen:
        file1_start = np.random.randint(0, file1_len - minlen)
        file1_end = file1_start + minlen 
        audio1 = audio1[file1_start:file1_end]

    if file2_len > minlen:
        file2_start = np.random.randint(0, file2_len - minlen)
        file2_end = file2_start + minlen
        audio2 = audio2[file2_start:file2_end]

    gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
    audio1 = rescale(audio1, torch.tensor(len(audio1)), gain, scale="dB")
    first_lvl = gain

    gain = np.clip(gain + random.normalvariate(-2.51, 2.66), -45, 0)
    audio2 = rescale(audio2, torch.tensor(len(tmp)), gain, scale="dB")

    mixture = audio1 + audio2
    max_amp = max(torch.abs(mixture).max().item(), torch.abs(audio1).max(dim=-1)[0].item())
    max_amp = max(max_amp, torch.abs(audio2).max(dim=-1)[0].item())

    mix_scaling = 1 / max_amp * 0.9
    sources = sources * mix_scaling
    mixture = mix_scaling * mixture
    return mixture

def make_scheduler(conf):    
    scheduler = None
    if conf["main_args"]["model"] in ["DPTNet", "SepFormerTasNet", "SepFormer2TasNet"]:
        steps_per_epoch = len(train_loader) // conf["main_args"]["accumulate_grad_batches"]
        conf["scheduler"]["steps_per_epoch"] = steps_per_epoch
        scheduler = {
            "scheduler": DPTNetScheduler(
                optimizer=optimizer,
                steps_per_epoch=steps_per_epoch,
                d_model=model.masker.mha_in_dim,
            ),
            "interval": "batch",
        }
    elif conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=5)
    return scheduler

def save_log(conf):
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)
    loggers = []
    tb_logger = pl.loggers.TensorBoardLogger(
        os.path.join(exp_dir, "tb_logs/"),
    )
    loggers.append(tb_logger)
    if conf["main_args"]["comet"]:
        comet_logger = pl.loggers.CometLogger(
            save_dir=os.path.join(exp_dir, "comet_logs/"),
            experiment_key=conf["main_args"].get("comet_exp_key", None),
            log_code=True,
            log_graph=True,
            parse_args=True,
            log_env_details=True,
            log_git_metadata=True,
            log_git_patch=True,
            log_env_gpu=True,
            log_env_cpu=True,
            log_env_host=True,
        )
        comet_logger.log_hyperparams(conf)
        loggers.append(comet_logger)


def make_callbacks(exp_dir, checkpoint_dir, conf):
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir, filename='{epoch}-{step}', monitor="val_loss", mode="min",
        save_top_k=conf["training"]["epochs"], save_last=True, verbose=True,
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=30, verbose=True))
    return callbacks
# def dynamic_mix_data_prep(hparams):

#     # 1. Define datasets
#     train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
#         csv_path=hparams["train_data"],
#         replacements={"data_root": hparams["data_folder"]},
#     )

#     # we build an dictionary where keys are speakers id and entries are list
#     # of utterances files of that speaker

#     spk_hashtable, spk_weights = build_spk_hashtable(hparams)

#     spk_list = [x for x in spk_hashtable.keys()]
#     spk_weights = [x / sum(spk_weights) for x in spk_weights]

#     @sb.utils.data_pipeline.takes("mix_wav")
#     @sb.utils.data_pipeline.provides("mix_sig", "s1_sig", "s2_sig")
#     def audio_pipeline(
#         mix_wav,
#     ):  # this is dummy --> it means one epoch will be same as without dynamic mixing

#         speakers = np.random.choice(
#             spk_list, hparams["num_spks"], replace=False, p=spk_weights
#         )
#         # select two speakers randomly
#         sources = []
#         first_lvl = None

#         spk_files = [
#             np.random.choice(spk_hashtable[spk], 1, False)[0]
#             for spk in speakers
#         ]

#         minlen = min(
#             *[torchaudio.info(x).num_frames for x in spk_files],
#             hparams["training_signal_len"],
#         )

#         for i, spk_file in enumerate(spk_files):

#             # select random offset
#             length = torchaudio.info(spk_file).num_frames
#             start = 0
#             stop = length
#             if length > minlen:  # take a random window
#                 start = np.random.randint(0, length - minlen)
#                 stop = start + minlen

#             tmp, fs_read = torchaudio.load(
#                 spk_file, frame_offset=start, num_frames=stop - start,
#             )

#             # peak = float(Path(spk_file).stem.split("_peak_")[-1])
#             tmp = tmp[0]  # * peak  # remove channel dim and normalize

#             if i == 0:
#                 gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
#                 tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
#                 # assert not torch.all(torch.isnan(tmp))
#                 first_lvl = gain
#             else:
#                 gain = np.clip(
#                     first_lvl + random.normalvariate(-2.51, 2.66), -45, 0
#                 )
#                 tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
#                 # assert not torch.all(torch.isnan(tmp))
#             sources.append(tmp)

#         # we mix the sources together
#         # here we can also use augmentations ! -> runs on cpu and for each
#         # mixture parameters will be different rather than for whole batch.
#         # no difference however for bsz=1 :)

#         # padding left
#         # sources, _ = batch_pad_right(sources)

#         sources = torch.stack(sources)
#         mixture = torch.sum(sources, 0)
#         max_amp = max(
#             torch.abs(mixture).max().item(),
#             *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
#         )
#         mix_scaling = 1 / max_amp * 0.9
#         sources = sources * mix_scaling
#         mixture = mix_scaling * mixture

#         yield mixture
#         for i in range(hparams["num_spks"]):
#             yield sources[i]

#     sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline)
#     sb.dataio.dataset.set_output_keys(
#         [train_data], ["id", "mix_sig", "s1_sig", "s2_sig"]
#     )

#     train_data = torch.utils.data.DataLoader(
#         train_data,
#         batch_size=hparams["dataloader_opts"]["batch_size"],
#         num_workers=hparams["dataloader_opts"]["num_workers"],
#         collate_fn=PaddedBatch,
#         worker_init_fn=lambda x: np.random.seed(
#             int.from_bytes(os.urandom(4), "little") + x
#         ),
#     )
#     return train_data

def parse_arguments(arg_list):
    """
    Parse command-line arguments to the experiment.
    Arguments
    ---------
    arg_list : list
        A list of arguments to parse, most often from 'sys.argv[1:]'.

    Returns
    --------
    param_file : str
        The location of the parameters file.
    run_opts : dict
        Run options, such as distributed, device, etc.
    overrides : dict
        The overrides to pass to ``load_hyperpyyaml``.

    Example
    --------
    >>> argv = ['hyperparams.yaml', '--device', 'cuda:1', '--seed', '10']
    >>> filename, run_opts, overrides, hparams = parse_arguments(argv) or parse_arguments(sys.argv[1:]) for .sh file
    >>> filename
    'hyperparams.yaml'
    >>> run_opts["device"]
    'cuda:1'
    >>> overrides
    'seed: 10'
    >>> hparams
    XX
    XX
    """

    parser = argparse.ArgumentParser(description = "Run a experiment")
    
    parser.add_argument("param_file", type = str, help = help="A yaml-formatted file using the extended YAML syntax defined by SpeechBrain")
    
    parser.add_argument("--debug", default=False, action ="store_true", help = "Run the experiment with only a few batches for all datasets, to ensure code runs without crashing.")

    parser.add_argument("--noprogressbar", default=False,action="store_true", help="This flag disables the data loop progressbars.")
    
    # parser.add_argument("--log_config", type=str, help="A file storing the configuration options for logging")
  
    # parser.add_argument("--local_rank", type=int, help="Rank on local machine")



    # parser.add_argument("--device", type=str, default="cuda:0", help="The device to run the experiment on (e.g. 'cuda:0')")
    # parser.add_argument("--data_parallel_count", type=int, default=-1,help="Number of devices that are used for data_parallel computation")
    #parser.add_argument( "--data_parallel_backend", default=False,action="store_true", help="This flag enables training with data_parallel.")

    # parser.add_argument("--distributed_launch", default=False,action="store_true", help="This flag enables training with DDP. Assumes script run with `torch.distributed.launch`")    )
    # parser.add_argument("--distributed_backend", type=str, default="nccl", help="One of {nccl, gloo, mpi}")

    # parser.add_argument("--jit_module_keys", type=str, nargs="*", help="A list of keys in the 'modules' dict to jitify")
    # parser.add_argument("--auto_mix_prec", default=False action="store_true", help="This flag enables training with automatic mixed-precision.")
    # parser.add_argument("--max_grad_norm", type=float, help="Gradient norm will be clipped to this value, enter negative value to disable.")
    # parser.add_argument("--nonfinite_patience", type=int, help="Max number of batches per epoch to skip if loss is nonfinite.")
    # parser.add_argument("--ckpt_interval_minutes", type=float,help="Amount of time between saving intra-epoch checkpoints in minutes. If non-positive, intra-epoch checkpoints are not saved.")

    known_args = parser.parse_known_args()[0]
    if known_args.debug == True:
        parser.add_argument("--debug_batches", type=int, default=2 help="Number of batches to run in debug mode.")
        parser.add_argument("--debug_epochs", type=int, default=2,
        help="Number of epochs to run in debug mode. If a non-positive number is passed, all epochs are run.")

    # Accept extra args to overrie yaml
    run_opts, overrides = parser.parse_known_args(arg_list)

    # Ignore items that are "None", they were not passed
    run_opts = {k: v for k, v in vars(run_opts).items() if v is not None}

    param_file = run_opts["param_file"]
    del run_opts["param_file"]

    overrides = _convert_to_yaml(overrides)

    # Checking that DataParallel use the right number of GPU
    if run_opts["data_parallel_backend"]:
        if run_opts["data_parallel_count"] == 0:
            raise ValueError(
                "data_parallel_count must be > 1."
                "if data_parallel_count = -1, then use all gpus."
            )
        if run_opts["data_parallel_count"] > torch.cuda.device_count():
            raise ValueError(
                "data_parallel_count must be <= "
                + str(torch.cuda.device_count())
                + "if data_parallel_count = -1, then use all gpus."
            )
    
    # For DDP, the device args must equal to local_rank used by
    # torch.distributed.launch. If run_opts["local_rank"] exists,
    # use os.environ["LOCAL_RANK"]
    local_rank = None
    if "local_rank" in run_opts:
        local_rank = run_opts["local_rank"]
    else:
        if "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "":
            local_rank = int(os.environ["LOCAL_RANK"])

    # force device arg to be the same as local_rank from torch.distributed.lunch
    if local_rank is not None and "cuda" in run_opts["device"]:
        run_opts["device"] = run_opts["device"][:-1] + str(local_rank)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    return param_file, run_opts, overrides, hparams


def trainer(conf):
    train_enh_dir = conf["main_args"].get("train_enh_dir", None)
    resume_ckpt = conf["main_args"].get("resume_ckpt", None)

    train_loader, val_loader, train_set_infos = make_dataloaders(
        corpus=conf["main_args"]["corpus"],
        train_dir=conf["data"]["train_dir"],
        val_dir=conf["data"]["valid_dir"],
        train_enh_dir=train_enh_dir,
        task=conf["data"]["task"],
        sample_rate=conf["data"]["sample_rate"],
        n_src=conf["data"]["n_src"],
        segment=conf["data"]["segment"],
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
    )   

    conf["masknet"].update({"n_src": conf["data"]["n_src"]})
    if conf["main_args"]["strategy"] == "multi_task":
        conf["masknet"].update({"n_src": conf["data"]["n_src"]+1})

    model = getattr(asteroid.models, conf["main_args"]["model"])(**conf["filterbank"], **conf["masknet"])

    optimizer = make_optimizer(model.parameters(), **conf["optim"])

    # Define scheduler
    scheduler = make_scheduler()


    # Define Loss function.
    pit_wrapper = MultiTaskLossWrapper if conf["main_args"]["strategy"] == "multi_task" else PITLossWrapper
    loss_func = pit_wrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    system = GeneralSystem(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    make_optimizer(exp_dir, checkpoint_dir, conf)


    # save the args and log
    save_log(conf)


    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None
    distributed_backend = "ddp" if torch.cuda.is_available() else None   


    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger() 

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        gpus=gpus,
        distributed_backend=distributed_backend,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        accumulate_grad_batches=conf["main_args"]["accumulate_grad_batches"],
        resume_from_checkpoint=resume_ckpt,
        deterministic=True,
        replace_sampler_ddp=False if conf["main_args"]["strategy"] == "multi_task" else True,
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set_infos)
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    

# https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch-lightning/Supercharge_your_Training_with_Pytorch_Lightning_%2B_Weights_%26_Biases.ipynb#scrollTo=A-N4UcuSD6Tx

pl.seed_everything(100)
hparams_file, run_opts, overrides, hparams = parse_arguments(sys.argv[1:)

create_experiment_directory(experiment_directory=hparams["output_folder"], hyperparams_to_save=hparams_file, overrides=overrides)

