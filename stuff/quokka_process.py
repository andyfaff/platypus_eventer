import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import h5py
import shutil
from pathlib import Path

from refnx.reduce import event
from platypus_eventer import analysis
from platypus_eventer.status import State, Status


def process_file(
        nx,
        frame_frequency,
        oscillation_period,
        subframe_bin_sz,
        nbins,
        pth,
        nef_pth,
        sef_pth
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

    print(f"{daq_dirname=}")

    # copy the NEF/SEF to here.
    shutil.copytree(
        str(nef_pth / daq_dirname) + "/",
        f"./{daq_dirname}",
        dirs_exist_ok=True
    )
    shutil.copytree(
        str(sef_pth / daq_dirname),
        f"./{daq_dirname}",
        dirs_exist_ok=True
    )

    # find out the maximum number of frames in the NEF
    s = Status()
    state = State(s.from_file(daq_dirname))
    max_frames = state.dct["current_frame"]

    # read SEF events into a list
    events = analysis.read_events(daq_dirname)

    # calculate which SEF frame corresponds to the first NEF
    frame_offset = analysis.predicted_frame(daq_dirname)

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
    # time corresponding to each of the T0 frames.
    t_sample = np.array([i[1] for i in events if i[2] == 1]) / 1e9

    # the T0 frame numbers
    f_t0 = np.array([i[0] for i in events if i[2] < 0])
    # time corresponding to each of the T0 frames.
    t_t0 = np.array([i[1] for i in events if i[2] < 1]) / 1e9

    # correct the SEF frame number so it lines up with the NEF
    f_t0 -= frame_offset
    f_sample -= frame_offset

    # identify any glitches caused by missed T0 pulses
    bf = analysis.identify_glitch(t_t0, 25)
    if np.count_nonzero(bf):
        _o = analysis.deglitch(f_t0, t_t0, f_sample, t_sample, volts, 25)
        f_t0, t_t0, f_sample, t_sample, volts = _o

    # work out fractional location for when the voltage was measured within the frame
    # This is necessary if we want to get precision much smaller than the frame period.
    _period = 1 / frame_frequency

    f_sample_frac = np.zeros_like(f_sample, dtype=np.float64)
    for i, _f_sample in enumerate(f_sample):
        _t_sample = t_sample[i]
        _idx = np.argwhere(f_t0 == _f_sample)[0, 0]
        _t_t0 = t_t0[_idx]
        f_sample_frac[i] = ((_t_sample - _t_t0)) / _period

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

    # fit a sine wave to the frame vs voltage information
    # initial guesses
    offset = np.mean(volts)
    amplitude = 0.5 * (np.max(volts) - np.min(volts))
    f0 = 11
    osc_period = oscillation_period

    # initial guess vector
    p0 = [offset, amplitude, f0, osc_period]

    res = differential_evolution(
        chi2,
        bounds=[
            (0.9 * offset, 1.1 * offset),
            (0.9 * amplitude, 1.1 * amplitude),
            (-15, 15),
            (0.9 * oscillation_period, 1.1 * oscillation_period),
        ],
        args=(f_sample + f_sample_frac, volts),
    )
    offset, amplitude, f0, osc_period = p0 = res.x
    oscillation_period = osc_period

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

    # load in the NEF
    with open(f"{daq_dirname}/DATASET_0/EOS.bin", "rb") as f:
        nef = event.events(f)
        f_events, t_events, y_events, x_events = nef[0]
        t_events = np.asarray(t_events, np.uint32)
        y_events = np.asarray(y_events, np.uint32)
        x_events = np.asarray(x_events, np.uint32)

        # histogram/rebin the TOF data according to the time_bins
        # t_events will be a number in [0, len(TIME_BINS)). i.e.
        # an array index for which time bin the neutron is in.
        t_events = np.digitize(t_events, TIME_BINS) - 1
        _events = np.c_[f_events, t_events, y_events, x_events]

    # Work out phase for each frame/subframe.
    # Should have shape (N, T) where N is the total number of frames
    # and T is the number of time bins/subframes.

    _mid_subframe_bins = 0.5 * (TIME_BINS[1:] + TIME_BINS[:-1]) / _period
    _frac_frames = f_t0[:, None] + _mid_subframe_bins[None, :]

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
    frame_phases = (phase(_frac_frames, p0) - _phase_offset) % (2 * np.pi)

    # Calculate where the phase of each of frames/subframes would land in the phase_bins.
    # i.e. it specifies which detector image each frame/subframe each neutron will end up in.
    bin_loc = np.digitize(frame_phases, phase_bins) - 1

    # make the detector image, N, Y, X
    detector = np.zeros((nbins, 192, 192), dtype=np.uint32)

    # bin each neutron event into the detector image
    # Remember that the tof information of the neutron events have already been digitised.
    # i.e. t refers to the index of which time bin/subframe the event belongs to.
    for i in _events:
        f, t, y, x = i
        det_idx = bin_loc[f, t]
        detector[det_idx, y, x] += 1

    assert np.sum(detector) == len(_events)
    frame_count_fraction = [np.count_nonzero(bin_loc==i) for i in range(nbins)] / np.prod(bin_loc.shape)

    qkk_patcher(nxfile, detector, frame_count_fraction, fpth)



def qkk_patcher(nxfile, detector, frame_count_fraction, pth):
    """
    Parameters
    ----------
    nxfile : str
    detector : np.ndarray
    frame_count_fraction : np.ndarray
    pth : Path-like
    """

    for i in range(len(detector)):
        new_nxfile = f"QKK{i:07d}"
        shutil.copy(
            pth / nxfile + ".nx.hdf",
            new_nxfile + ".nx.hdf"
        )

        with h5py.File(f"{new_nxfile}.nx.hdf", "r+") as f:
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
