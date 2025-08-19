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
            0.9 * window * period < t_f[i + window] - t_f[i] < 1.1 * window * period
        ):
            glitch[i] = True
    return glitch


def deglitch(f_f, t_f, f_v, t_v, voltage, frame_frequency, window=5):
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

    Returns
    -------
    f_f, t_f, f_v, t_v: tuple
        De-glitched sample environment event stream
    """
    period = 1 / frame_frequency
    while True:

        # identify bad frames
        g = identify_glitch(t_f, frame_frequency, window=window)
        if not np.sum(np.count_nonzero(g)):
            break

        bad_frames = np.argwhere(g)
        first_bad = last_bad = bad_frames[0, 0]
        # how long do the bad frames last
        for i in range(1, bad_frames.size):
            if bad_frames[i, 0] == last_bad + 1:
                last_bad += 1
            else:
                break
        window_size = last_bad - first_bad + 1

        # need to get rid of first_bad:first_bad + window_size
        last_good_before_glitch = first_bad - 1
        first_good_after_glitch = first_bad + window_size

        num_new_frames = (
            int(
                np.round(
                    (t_f[first_good_after_glitch] - t_f[last_good_before_glitch])
                    / period
                )
            )
            - 1
        )
        # print(f"{num_new_frames=}, {first_bad=}, {window_size=}")

        # remove bad frames
        f_f = np.delete(f_f, slice(first_bad, first_good_after_glitch))
        f_f[first_bad:] += num_new_frames - window_size

        new_times = np.linspace(
            t_f[last_good_before_glitch] + period,
            t_f[first_good_after_glitch] - period,
            num_new_frames,
        )
        t_f = np.delete(t_f, slice(first_bad, first_good_after_glitch))

        # find which voltage measurements to delete
        _idx = np.argwhere(
            np.logical_and(first_bad <= f_v, f_v < first_good_after_glitch)
        )
        f_v = np.delete(f_v, _idx)
        t_v = np.delete(t_v, _idx)
        voltage = np.delete(voltage, _idx)
        # renumber voltage frames after the glitch
        _idx = np.argwhere(first_good_after_glitch <= f_v)
        f_v[_idx] += num_new_frames - window_size

        # now put replacement frames in.
        new_frames = np.arange(first_bad, first_bad + num_new_frames)

        f_f = np.insert(f_f, first_bad, new_frames)
        t_f = np.insert(t_f, first_bad, new_times)

    return f_f, t_f, f_v, t_v, voltage
