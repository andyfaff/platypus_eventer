import gzip
import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from status import Status, parse_status
from refnx.reduce.event import events
from streamer import _struct


def _event_sef(buf):
    # get a list of events from the SEF based on a binary buffer
    it = struct.iter_unpack(_struct, buf)
    return list(it)


def _predicted_frame(ev, dataset_start_time_t):
    # figures out which frame in the SEF corresponds to the first frame
    # in the NEF
    t0 = [ev[i][1] for i in range(len(ev)) if ev[i][-1] == b'\x00~']

    idx = np.searchsorted(np.array(t0) / 1e9, dataset_start_time_t)
    nearest_time = np.argmin(((np.array(t0)) / 1e9 - dataset_start_time_t) ** 2)
    print(f"array search: {int(idx)}, nearest time: {nearest_time}")
    return nearest_time


def predicted_frame(daq_dirname, dataset=0, pth="."):
    s = Status("manager")
    loc = Path(pth) / daq_dirname / f"DATASET_{dataset}"
    status = parse_status(s.from_file(daq_dirname, dataset=dataset, pth=pth))[0]
    dataset_start_time_t = status['dataset_start_time_t']

    with gzip.GzipFile(str(loc / "EOS.gz"), 'rb') as f:
        buf = f.read(1_000_000_000)

    _sef_events = _event_sef(buf)
    offset = _predicted_frame(_sef_events, dataset_start_time_t)
    return offset
