import gzip
import numpy as np
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from status import Status, State
from refnx.reduce.event import events
from streamer import _struct, _struct_sz


def _event_sef(buf):
    # get a list of events from the SEF based on a binary buffer
    it = struct.iter_unpack(_struct, buf)
    return list(it)


def _predicted_frame(ev, dataset_start_time_t):
    # figures out which frame in the SEF corresponds to the first frame
    # in the NEF
    t0 = [ev[i][1] for i in range(len(ev)) if ev[i][-1] == b"\x00~"]

    idx = np.searchsorted(np.array(t0) / 1e9, dataset_start_time_t)
    nearest_time = np.argmin(((np.array(t0)) / 1e9 - dataset_start_time_t) ** 2)
    print(f"array search: {int(idx)}, nearest time: {nearest_time}")
    return nearest_time


def predicted_frame(daq_dirname, dataset=0, pth="."):
    s = Status("manager")
    loc = Path(pth) / daq_dirname / f"DATASET_{dataset}"
    state = State(s.from_file(daq_dirname, dataset=dataset, pth=pth))[0]
    dataset_start_time_t = state.dataset_start_time_t

    _events = []
    with gzip.GzipFile(str(loc / "EOS.gz"), "rb") as f:
        while True:
            buf = f.read(_struct_sz * 1_092)
            if not buf:
                break
            _sef_events = _event_sef(buf)
            _events.extend(_sef_events)
    offset = _predicted_frame(_events, dataset_start_time_t)
    return offset
