import gzip
import numpy as np
import struct
from pathlib import Path
from platypus_eventer.status import Status, State
from refnx.reduce.event import events
from platypus_eventer.streamer import _struct, _struct_sz


def _event_sef(buf):
    # get a list of events from the SEF based on a binary buffer
    it = struct.iter_unpack(_struct, buf)
    return list(it)


def _predicted_frame(ev, dataset_start_time_t):
    # figures out which frame in the SEF corresponds to the first frame
    # in the NEF
    t0 = [ev[i][1] for i in range(len(ev)) if ev[i][2] == -1]

    idx = np.searchsorted(np.array(t0) / 1e9, dataset_start_time_t)
    nearest_time = np.argmin(((np.array(t0)) / 1e9 - dataset_start_time_t) ** 2)
    print(f"array search: {int(idx)}, nearest time: {nearest_time}")
    return nearest_time


def read_events(daq_dirname, dataset=0, pth="."):
    """
    Reads the Sample Event File from the daq_dirname directory.

    Parameters
    ----------
    daq_dirname : {str, Path-like}
        location of the SEF file
    dataset : int
        which dataset entry to read
    pth : {str, Path-like}
        directory where daq_dirname is located

    Returns
    -------
    events: list of tuples
        Each tuple contains a single event. The tuple contains:
        (frame, time, channel, voltage)

        `time` is specified in ns since the epoch at which the event occurred.
        `channel` is positive for a voltage measurement, negative for
        a T0 event.
        `voltage` is the voltage measurement measured on the ADC `channel`.
    """
    loc = Path(pth) / daq_dirname / f"DATASET_{dataset}"
    _events = []
    with gzip.GzipFile(str(loc / "EOS.gz"), "rb") as f:
        while True:
            buf = f.read(_struct_sz * 1_092)
            if not buf:
                break
            _sef_events = _event_sef(buf)
            _events.extend(_sef_events)
    return _events


def predicted_frame(daq_dirname, dataset=0, pth="."):
    s = Status()
    state = State(s.from_file(daq_dirname, dataset=dataset, pth=pth))
    dataset_start_time_t = state.dataset_start_time_t

    _events = read_events(daq_dirname, dataset=dataset, pth=pth)

    offset = _predicted_frame(_events, dataset_start_time_t)
    return offset


def identify_glitch(t_f, frame_frequency, window=5):
    """
    Identifies any glitches in the data acquisition caused by
    missed T0 pulses.

    Parameters
    ----------
    t_f: array-like
        time of T0 pulses
    frame_frequency: float
        frequency of T0 signal generation
    window: int
        spread of pulses to check for glitches

    Returns
    -------
    glitches: array-like
        Specifies which sets of pulses have glitches

    Glitch identification works by running a window across all the pulses and checking
    that the apparent time separation for that number of pulses is close
    (currently 10%) to a multiple of the T0 pulse period.
    """
    window = int(window)
    period = 1 / frame_frequency
    npnts = len(t_f)

    glitch = np.zeros((npnts,), np.bool)
    for i in range(npnts - window):
        if not (
            window * period - (period * 0.5)
            < t_f[i + window] - t_f[i]
            < window * period + (period * 0.5)
        ):
            glitch[i] = True
    return glitch


def deglitch(f_f, t_f, f_v, t_v, voltage, frame_frequency, window=5, debug=False):
    """
    Removes glitches caused by missed T0 pulses in a Sample Event File
    train. Glitches in predicted frame times are replaced with model
    values based on the T0 period.

    Parameters
    ----------
    f_f: array-like
        frame number of T0 pulses
    t_f: array-like
        time of T0 pulses (seconds)
    f_v: array-like
        frame when sample environment voltage was measured
    t_v: array-like
        time of when sample environment voltage was measured
    voltage: array-like
        measured sample environment voltage
    frame_frequency: float
        frequency of T0 signal generation
    window: int
        spread of pulses to check for glitches
    debug: bool
        Check glitches removed correctly

    Returns
    -------
    f_f, t_f, f_v, t_v, voltage: tuple
        De-glitched sample environment event stream
    """
    period = 1 / frame_frequency
    glitches = (
        np.argwhere(identify_glitch(t_f, frame_frequency, window=window)).flatten() + 1
    )

    if not len(glitches):
        # no glitches
        return f_f, t_f, f_v, t_v, voltage

    # collect the dodgy frames in batches
    _diff = np.diff(glitches) > 1
    batches = []
    batch = []
    for i, _d in enumerate(_diff):
        if _d:
            batches.append(np.r_[np.array(batch), glitches[i], glitches[i] + 1])
            batch = []
        else:
            batch.append(glitches[i])

    batches.append(np.r_[np.array(batch), glitches[-1], glitches[-1] + 1])

    # now remove frames from the batches
    new_t_f = []
    start_good = 0
    inserted = []
    for batch in batches:
        start_bad = batch[0]
        new_t_f.append(t_f[start_good:start_bad])
        time_diff = t_f[batch[-1] + 1] - t_f[start_bad - 1]
        npnts = int(round(time_diff / period)) - 1
        inserted.append(npnts)
        new_times = np.linspace(
            t_f[start_bad - 1] + period,
            t_f[batch[-1] + 1] - period,
            num=npnts,
            endpoint=True,
        )
        new_t_f.append(new_times)

        start_good = batch[-1] + 1

    new_t_f.append(t_f[batches[-1][-1] + 1 :])

    new_t_f = np.concatenate(new_t_f)
    new_f_f = np.arange(f_f[0], f_f[0] + len(new_t_f))

    # now adjust f_v, t_v
    for i in reversed(range(len(batches))):
        batch = batches[i]
        insert = inserted[i]
        l = np.searchsorted(f_v, batch[0])
        r = np.searchsorted(f_v, batch[-1], side="right")

        f_v = np.delete(f_v, slice(l, r))
        f_v[l:] += -len(batch) + insert
        t_v = np.delete(t_v, slice(l, r))
        voltage = np.delete(voltage, slice(l, r))

    if debug:
        check_sample_times(new_f_f, new_t_f, f_v, t_v)
    return new_f_f, new_t_f, f_v, t_v, voltage


def check_sample_times(f_f, t_f, f_v, t_v):
    for i, (_f_v, _t_v) in enumerate(zip(f_v, t_v)):
        loc = np.searchsorted(f_f, _f_v)
        try:
            assert t_f[loc] < _t_v < t_f[loc + 1]
        except AssertionError as e:
            print(f"{i=}, {loc=}, {_f_v=},  {t_f[loc]=}, {_t_v=},  {t_f[loc+1]=}")
            raise e
