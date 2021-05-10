from torch.utils.data import DataLoader
# from data import LibriMix, WhamDataset

from .utils import MultiTaskDataLoader

from .avspeech_dataset import AVSpeechDataset
from .wham_dataset import WhamDataset
from .whamr_dataset import WhamRDataset
from .dns_dataset import DNSDataset
from .librimix_dataset import LibriMix
from .wsj0_mix import Wsj0mixDataset
from .musdb18_dataset import MUSDB18Dataset
from .sms_wsj_dataset import SmsWsjDataset
from .kinect_wsj import KinectWsjMixDataset
from .fuss_dataset import FUSSDataset
from .dampvsep_dataset import DAMPVSEPSinglesDataset

__all__ = [
    "AVSpeechDataset",
    "WhamDataset",
    "WhamRDataset",
    "DNSDataset",
    "LibriMix",
    "Wsj0mixDataset",
    "MUSDB18Dataset",
    "SmsWsjDataset",
    "KinectWsjMixDataset",
    "FUSSDataset",
    "DAMPVSEPSinglesDataset",
]


def make_dataloaders(corpus, train_dir, val_dir, train_enh_dir=None, task="sep_clean", sample_rate=8000, n_src=2, segment=4.0, batch_size=4, num_workers=None,):
    if corpus == "LibriMix":
        train_set = LibriMix(csv_dir=train_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=segment,)
        val_set = LibriMix(csv_dir=val_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=segment,)
    elif corpus == "wsj0-mix":
        train_set = WhamDataset(json_dir=train_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=segment,)
        val_set = WhamDataset(json_dir=val_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=segment,)

    if train_enh_dir is None:
        train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,)
    else:
        train_enh_set = LibriMix(csv_dir=train_enh_dir, task="enh_single", sample_rate=sample_rate, n_src=1, segment=segment,)
        train_loader = MultiTaskDataLoader([train_set, train_enh_set],
                                           shuffle=True, batch_size=batch_size, drop_last=True, num_workers=num_workers,)
    val_loader = DataLoader(val_set, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=True,)
    
    infos = train_set.get_infos()    
    return train_loader, val_loader, infos


def make_test_dataset(corpus, test_dir, task="sep_clean", sample_rate=8000, n_src=2):
    if corpus == "LibriMix":
        test_set = LibriMix(csv_dir=test_dir, task=task, sample_rate=sample_rate, n_src=n_src, segment=None,)
    elif corpus == "wsj0-mix":
        test_set = WhamDataset(json_dir=test_dir, task=task, sample_rate=sample_rate, nondefault_nsrc=n_src, segment=None,)
    return test_set