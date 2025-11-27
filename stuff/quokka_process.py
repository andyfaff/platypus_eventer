import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, minimize
import h5py
import time
import shutil
from pathlib import Path

from refnx.reduce import event
from platypus_eventer import analysis
from platypus_eventer.status import State, Status


import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import h5py
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

from refnx.reduce import event
from platypus_eventer import analysis
from platypus_eventer.status import State, Status


def wave(x, p):
    offset, amplitude, f0, osc_period = p
    return offset + amplitude * np.sin(phase(x, p))


def phase(x, p):
    x = np.asarray(x, np.float64)
    offset, amplitude, f0, osc_period = p
    return 2 * np.pi * (x - f0) / osc_period


def chi2(p, *args):
    x = args[0]
    y = args[1]
    ideal = wave(x, p)
    return np.sum((y - ideal) ** 2)


def event_reader(binfile, numframes=7_500):
    # 7500 frames is 5 minutes at 25 Hz
    fcount = 0
    eole = [127]
    with open(binfile, "rb") as fi:
        while True:
            ev, eole = event.events(fi, end_last_event=eole[-1], max_frames=numframes)
            if len(ev[0]) > 0:
                f, t, y, x = ev
                f += fcount
                yield (f, t, y, x)
                fcount += numframes
            else:
                break


def process_file(
    nx,
    frame_frequency,
    oscillation_period,
    subframe_bin_sz,
    nbins,
    pth,
    nef_pth,
    sef_pth,
    collect_time=None,
):
    """
    Parameters
    ----------
    NX: int
        number of nexus file
    frame_frequency: float
        Frequency of T0 pulse (Hz)
    oscillation_period: float
        Oscillation period or sample environment, expressed in number of frames
        i.e. how many T0 frames does it take for one oscillation of sample env.
    subframe_bin_sz: float
       How finely we divide each frame. The frame period (1/frame_frequency) should
       be exactly divisible by this number. This value controls the sampling rate.
       Expressed in ms!
    nbins: int
        How finely would you like to slice up the waveform (i.e. how many scattering
        patterns in [0, 2*pi]?
    pth: Path-like
        Path to NeXUS file
    nef_pth: Path-like
        Path to NEF file
    sef_pth: Path-like
        Path to SEF file
    collect_time: None or (float, float)
        If specified, the processing discards neutron data outside this time period.
        (180, 1000) would include data between 180 and 1000 seconds.
    """
    nxfile = f"QKK{nx:07d}"
    fpth = Path(pth)

    # grab the file
    with h5py.File(fpth / f"{nxfile}.nx.hdf", "r") as fi:
        _pth = f"/{nxfile}/instrument/detector/daq_dirname"
        daq_dirname = fi[_pth][:][0].decode()
        _pth = f"/{nxfile}/instrument/parameters/L2"
        detector_y = fi[_pth][0]
        _pth = f"/{nxfile}/instrument/velocity_selector/wavelength_nominal"
        lamda = fi[_pth][0]
        _pth = f"/{nxfile}/instrument/detector/time"
        measurement_time = fi[_pth][0]

    print(f"Processing {str(fpth / f"{nxfile}.nx.hdf")}, {daq_dirname=}")

    # copy the NEF/SEF to here.
    print("Obtaining NEF/SEF and merging")
    shutil.copytree(
        str(nef_pth / daq_dirname) + "/", f"{nx}/{daq_dirname}", dirs_exist_ok=True
    )
    shutil.copytree(
        str(sef_pth / daq_dirname), f"{nx}/{daq_dirname}", dirs_exist_ok=True
    )

    # find out the maximum number of frames in the NEF
    s = Status()
    state = State(s.from_file(f"{nx}/{daq_dirname}"))
    max_frames = state.dct["current_frame"]

    print("loading SEF")
    # read SEF events into a list
    events = analysis.read_events(f"{nx}/{daq_dirname}")

    # calculate which SEF frame corresponds to the first NEF
    frame_offset = analysis.predicted_frame(f"{nx}/{daq_dirname}")

    # remove all events less than frame_offset
    events = [i for i in events if frame_offset <= i[0]]

    # extract data from the SEF
    # meaning of columns:
    # frame_no, time (ns), channel, voltage
    # channel > 0 if entry is an ADC channel measuring the voltage
    # channel < 0 if entry is a T0 pulse being recorded

    # extract all the voltage measurement events
    volts = [i[-1] for i in events if i[2] == 1]
    volts = np.array(volts)
    # the frame number corresponding to the voltage measurement events
    f_sample = np.array([i[0] for i in events if i[2] == 1])
    # time corresponding to each of the voltage measurement events
    t_sample = np.array([i[1] for i in events if i[2] == 1]) / 1e9

    # the T0 frame numbers
    f_t0 = np.array([i[0] for i in events if i[2] < 0])
    # time corresponding to each of the T0 frames.
    t_t0 = np.array([i[1] for i in events if i[2] < 0]) / 1e9

    # correct the SEF frame number so it lines up with the NEF
    f_t0 -= frame_offset
    f_sample -= frame_offset

    # identify any glitches caused by missed T0 pulses
    print("deglitching")
    bf = analysis.identify_glitch(t_t0, 25)
    if np.count_nonzero(bf):
        _o = analysis.deglitch(f_t0, t_t0, f_sample, t_sample, volts, 25)
        f_t0, t_t0, f_sample, t_sample, volts = _o

    # perhaps we only want to use a subset of the data
    start_frame = 0
    end_frame = np.inf
    fractional_time = 1
    if collect_time is not None:
        start_frame = max(collect_time[0] * frame_frequency, 0)
        end_frame = collect_time[1] * frame_frequency
        # what fraction of the measurement did we want to examine?
        fractional_time = (collect_time[1] - collect_time[0]) / measurement_time
        if collect_time[1] > measurement_time:
            raise ValueError("collect_time[1] is greater than the measurement_time")

    # remove SEF frames that we're not interested in.
    _subset = np.searchsorted(f_t0, [start_frame, end_frame])
    f_t0 = f_t0[_subset[0]:_subset[1]]
    t_t0 = t_t0[_subset[0]:_subset[1]]
    assert len(f_t0) == len(t_t0)
    _subset = np.searchsorted(f_sample, [start_frame, end_frame])
    f_sample = f_sample[_subset[0]:_subset[1]]
    t_sample = t_sample[_subset[0]:_subset[1]]
    volts = volts[_subset[0]:_subset[1]]
    assert len(f_sample) == len(t_sample) == len(volts)

    # work out fractional location for when the voltage was measured within the frame
    # This is necessary if we want to get precision much smaller than the frame period.
    _period = 1 / frame_frequency

    f_sample_frac = np.zeros_like(f_sample, dtype=np.float64)
    for i, _f_sample in enumerate(f_sample):
        _t_sample = t_sample[i]
        _idx = np.argwhere(f_t0 == _f_sample)[0, 0]
        _t_t0 = t_t0[_idx]
        f_sample_frac[i] = ((_t_sample - _t_t0)) / _period
        assert 0 < f_sample_frac[i] < 1

    # fit a sine wave to the frame vs voltage information
    # initial guesses
    offset = np.mean(volts)
    amplitude = 0.5 * (np.max(volts) - np.min(volts))
    f0 = 11
    osc_period = oscillation_period

    # initial guess vector
    p0 = [offset, amplitude, f0, osc_period]

    print("fitting sine wave")
    print(time.time())
    np.savetxt("fs.txt", f_sample + f_sample_frac)
    np.savetxt("vs.txt", volts)
    res = differential_evolution(
        chi2,
        bounds=[
            (0.9 * offset, 1.1 * offset),
            (0.9 * amplitude, 1.1 * amplitude),
            (0, oscillation_period),
            (0.9 * oscillation_period, 1.1 * oscillation_period),
        ],
        args=(f_sample + f_sample_frac, volts),
        # workers=-1,
        # polish=False
    )
    offset, amplitude, f0, osc_period = p0 = res.x
    np.testing.assert_allclose(p0, res.x)
    oscillation_period = osc_period
    print(f"{offset=}, {amplitude=}, {f0=}, {osc_period=}")
    print(time.time())
    
    fig, ax = plt.subplots(1)
    ax.plot(f_sample + f_sample_frac, volts);
    ax.plot(f_sample + f_sample_frac, wave(f_sample + f_sample_frac, res.x))
    ax.set_xlim(0, 100)
    fig.savefig("fit.png")

    # these specify the phases with the oscillation for which we wish to produce
    # scattering curves. *They are bin edges*.
    phase_bins = np.linspace(0, 2 * np.pi, nbins + 1)

    # Time bins for rebinning the TOF information within the NEF.
    # Necessary if we wish to sample at a higher rate.
    # These time bins control the sample period.
    # If the time bin width is 2000 us, then the sampling rate is 500 Hz.

    # From now on we use time bin/subframe interchangeably.
    _period = 1000000 / frame_frequency  # microseconds
    TIME_BINS = np.linspace(0, _period, int(_period / (1000 * subframe_bin_sz)) + 1)

    # Work out phase for each frame/subframe.
    # Should have shape (N, T) where N is the total number of frames
    # and T is the number of time bins/subframes.
    # _frac_frames is in Neutron Frame space
    _mid_subframe_bins = 0.5 * (TIME_BINS[1:] + TIME_BINS[:-1]) / _period
    _frac_frames = f_t0[:, None] + _mid_subframe_bins[None, :]
    assert (np.diff(f_t0) == 1).all()

    # Calculate the sine wave phase for each frame/subframe.
    # Note, these are predicted phases based on the fitted model above. If the
    # sine wave fit is poor, then the whole reduction falls over.

    # ADD/SUBTRACT CONSTANT PHASE_OFFSET THAT REPRESENT THE FLIGHT TIME OF THE NEUTRONS
    _neutron_speed = 3954 / lamda
    _flight_time = (detector_y / 1000) / _neutron_speed
    # proportion of frame that the flight time represents
    _p = _flight_time / (1 / frame_frequency)
    # proportion of SE oscillation that the flight time represents
    _phase_offset = _p / oscillation_period
    # convert to radians
    _phase_offset *= 2 * np.pi

    # phase for each frame/{subframe, time bin}
    # works out what sample phase each neutron frame/subframe corresponds to.
    frame_phases = (phase(_frac_frames, p0) - _phase_offset) % (2 * np.pi)

    # Calculate where the phase of each of frames/subframes would land in the phase_bins.
    # i.e. it specifies which detector image each frame/subframe each neutron will end up in.
    print("digitising frames")
    bin_loc = np.digitize(frame_phases, phase_bins) - 1
    # print(bin_loc[0:50]) # check that the sorting is incrementing correctly

    # make the detector image, N, Y, X
    detector = np.zeros((nbins, 192, 192), dtype=np.uint32)

    # bin neutron events into the detector image
    # tof information of the neutron events has to be digitised.
    # i.e. t refers to the index of which time bin/subframe the event belongs to.
    _event_reader = event_reader(f"{nx}/{daq_dirname}/DATASET_0/EOS.bin")

    _cts = 0
    for nef in _event_reader:
        f_events, t_events, y_events, x_events = nef
        t_events = np.asarray(t_events, np.uint32)
        y_events = np.asarray(y_events, np.uint32)
        x_events = np.asarray(x_events, np.uint32)

        # histogram/rebin the TOF data according to the time_bins
        # t_events will be a number in [0, len(TIME_BINS)). i.e.
        # an array index for which time bin the neutron is in.
        t_events = np.digitize(t_events, TIME_BINS) - 1
        _events = np.c_[f_events, t_events, y_events, x_events]

        # figure out which events to process
        strt_idx = 0
        end_idx = len(_events)
        if f_events[-1] < start_frame:
            # we don't need any events from this read iteration, read some more.
            continue
        elif f_events[0] >= end_frame:
            # the first frame in this read iteration is already at the end number of frames
            break

        if f_events[0] <= start_frame <= f_events[-1]:
            strt_idx = np.searchsorted(f_events, start_frame)
        if f_events[0] <= end_frame <= f_events[-1]:
            end_idx = np.searchsorted(f_events, end_frame)

        for i in _events[strt_idx:end_idx]:
            f, t, y, x = i
            # assert f in f_t0
            try:
                det_idx = bin_loc[f, t]
            except IndexError as e:
                # some TOF lie outside the timebins for some reason
                continue
            detector[det_idx, y, x] += 1

        if end_frame < f_events[-1]:
            # we've already read enough frames, we can stop reading the NEF
            break

    # ev = event.events(f"{nx}/{daq_dirname}/DATASET_0/EOS.bin")[0]
    # assert ev[0][-1] == f_events[-1]
    # assert np.sum(detector) == len(ev[0])

    # frame_count_fraction determines the amount of distributed monitor counts/time
    # for each synthesised dataset.
    if collect_time is not None:
        # trim down f_t0, bin_loc
        strt_idx = np.searchsorted(f_t0, start_frame)
        end_idx = np.searchsorted(f_t0, end_frame)
        bin_loc = bin_loc[strt_idx:end_idx]

    frame_count_fraction = [
        np.count_nonzero(bin_loc == i) for i in range(nbins)
    ] / np.prod(bin_loc.shape)
    print("patching file")
    qkk_patcher(nx, detector, frame_count_fraction * fractional_time, fpth)
    print("finished")


def qkk_patcher(nx, detector, frame_count_fraction, pth):
    """
    Parameters
    ----------
    nx: int
    detector : np.ndarray
    frame_count_fraction : np.ndarray
    pth : Path-like
    """
    nxfile = f"QKK{nx:07d}"
    dst_pth = Path(f"./{nx}")
    for i in range(len(detector)):
        new_nxfile = f"QKK{i:07d}"
        shutil.copy(pth / (nxfile + ".nx.hdf"), dst_pth / (new_nxfile + ".nx.hdf"))

        with h5py.File(dst_pth / f"{new_nxfile}.nx.hdf", "r+") as f:
            # rename group from the nxfile. For some reason Quokka is special
            # and the first child of the root node is named after the datafile
            f.move(f"/{nxfile}", f"/{new_nxfile}")

            # detector image
            dataset_name = f"/{new_nxfile}/data/hmm_xy"
            dset = f[dataset_name]
            dset[...] = detector[i]
            dset.attrs["target"] = f"/{new_nxfile}/instrument/detector/hmm_xy".encode()

            # total counts
            dataset_name = f"/{new_nxfile}/data/total_counts"
            dset = f[dataset_name]
            dset[...] = np.sum(detector[i])

            # total counts
            dataset_name = f"/{new_nxfile}/data/total_counts"
            dset = f[dataset_name]
            dset[...] = np.sum(detector[i])

            # measurement time
            dataset_name = f"/{new_nxfile}/instrument/detector/time"
            dset = f[dataset_name]
            t = dset[0]
            dset[...] = frame_count_fraction[i] * t

            ####################
            # beam monitor stuff
            ####################
            dataset_name = f"/{new_nxfile}/monitor/bm1_counts"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm

            dataset_name = f"/{new_nxfile}/monitor/bm1_time"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm

            dataset_name = f"/{new_nxfile}/monitor/bm2_counts"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm

            dataset_name = f"/{new_nxfile}/monitor/bm2_time"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm

            dataset_name = f"/{new_nxfile}/monitor/time"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm

            dataset_name = f"/{new_nxfile}/monitor/data"
            dset = f[dataset_name]
            bm = dset[0]
            dset[...] = frame_count_fraction[i] * bm
