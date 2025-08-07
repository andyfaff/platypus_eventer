from multiprocessing import Process, Queue, Value, Event
import multiprocessing as mp
import ctypes
import time
import configparser
from pathlib import Path
import os
import glob
import signal
import sys
from functools import partial
from status import Status, State
import streamer


class EventStreamer:
    """
    Class for starting/stopping streaming of sample events
    Creates child Processes for detecting the T0 frame pulse,
    measuring the sample events, and writing both events to file.
    """

    def __init__(self):
        self.stream_loc = None

        self.queue = None
        self.shutdown_event = Event()
        self.frame_event = Event()
        self.p_writer = None
        self.p_ADC_streamer = None
        self.p_t0_streamer = None

        self.currently_streaming = False
        self.frame = Value(ctypes.c_uint32)

    def stop(self, DATASET_number=None, final_state=None):
        self.currently_streaming = False
        self.shutdown_event.set()

        if self.p_t0_streamer is not None:
            # print("::::::::::::::::::::::::::::::::::::::::::::::")
            # print(self.p_t0_streamer)
            try:
                self.p_t0_streamer.join()
                self.p_t0_streamer.close()
            except ValueError:
                pass

        if self.p_ADC_streamer is not None:
            try:
                self.p_ADC_streamer.join()
                self.p_ADC_streamer.close()
            except ValueError:
                pass

        if self.queue is not None:
            self.queue.put(None)
            while not self.queue.empty():
                time.sleep(0.1)

        if self.p_writer is not None:
            try:
                self.p_writer.join()
                self.p_writer.close()
            except ValueError:
                pass

        if DATASET_number is not None and final_state is not None:
            parent = self.stream_loc.parent
            new_name = parent / f"DATASET_{DATASET_number}"
            self.stream_loc.rename(new_name)
            self.stream_loc = new_name
            with open(self.stream_loc / "final_state.txt", "w") as f:
                f.write(final_state)

    def start(self, stream_loc):
        self.stream_loc = stream_loc
        self.shutdown_event.clear()
        self.frame_event.clear()

        self.frame = Value(ctypes.c_int32, -1)
        self.queue = Queue()
        self.p_writer = Process(target=streamer.writer, args=(stream_loc, self.queue))
        self.p_t0_streamer = Process(
            target=streamer.T0_streamer,
            args=(self.frame, self.frame_event, self.queue, self.shutdown_event),
        )
        self.p_ADC_streamer = Process(
            target=streamer.ADC_streamer,
            args=(self.frame, self.frame_event, self.queue, self.shutdown_event),
        )

        self.p_writer.start()
        self.p_t0_streamer.start()
        self.p_ADC_streamer.start()

        self.currently_streaming = True


def _signal_close(streamer, signal, frame):
    if mp.parent_process() is None:
        streamer.stop()
        sys.exit()


def _create_stream_directory(pth, state, dataset_number_being_written=0):
    """
    Creates the streaming directory. Also saves the text status in the
    streaming directory

    Parameters
    ----------
    pth : Path
        Where to create the streaming directory
    state : State
        The state of the histogram server
    dataset_number_being_written : int
        Which dataset is currently being streamed

    Returns
    -------
    stream_loc : Path
        The path of the streaming directory
    """
    stream_loc = pth / state.DAQ_dirname / f"DATASET_{dataset_number_being_written}"
    stream_loc = stream_loc.resolve()
    os.makedirs(stream_loc, exist_ok=True)
    print(f"New sample event file starting, {stream_loc=}")
    with open(stream_loc / "state.txt", "w") as f:
        f.write(state.response)

    return stream_loc


def main(user="manager", password="", pth=None):
    """
    Starts the streaming process for sample events

    Parameters
    ----------
    user : str
        User name for the DAS server
    password: str
        Password for the DAS server
    pth : str
        Parent directory for all the streamed data
    """
    if pth is None:
        pth = Path.cwd()
    else:
        pth = Path(pth)

    config = configparser.ConfigParser()
    config.read(str(pth / "config.ini"))
    das_server = config.get("url", "das")

    s = Status(user=user, password=password, url=das_server)
    old_state = State(s())
    streamer = EventStreamer()

    # shutdown gracefully
    signal_close = partial(_signal_close, streamer)
    signal.signal(signal.SIGINT, signal_close)

    update_period = 2.0

    actual_dataset_number = 0
    dataset_number_being_written = 0
    LAST_DAQ_DIRNAME = ""
    STATE_REQUIRES_UPDATE = True

    while True:
        _s = s()
        state = State(_s)

        if state.started:
            actual_dataset_number = state.DATASET_number
            if actual_dataset_number != dataset_number_being_written:
                # we'll want to stop the streaming and update directories
                pass
            else:
                # dataset number is correct
                update_period = 2.0
            if STATE_REQUIRES_UPDATE:
                # Update state.txt file when the acquisition has finally started
                _create_stream_directory(
                    pth,
                    state,
                    dataset_number_being_written=dataset_number_being_written,
                )
                STATE_REQUIRES_UPDATE = False

        if streamer.currently_streaming and (
            not (state.started or state.starting)
            or state.DAQ_dirname != old_state.DAQ_dirname
            or (state.started and actual_dataset_number != dataset_number_being_written)
        ):
            # stop streaming
            print(f"Stopping streamer, {state.DAQ_dirname=}")
            streamer.stop(actual_dataset_number, _s)
            update_period = 1

        if (state.started or state.starting) and not streamer.currently_streaming:
            # time to start new directories and start streaming
            # textstatus does not fully update until the acquisition has started.
            if LAST_DAQ_DIRNAME != state.DAQ_dirname:
                # totally new dataset
                LAST_DAQ_DIRNAME = state.DAQ_dirname
                dataset_number_being_written = 0
            elif state.started:
                dataset_number_being_written = state.DATASET_number

            elif state.starting:
                # dataset number probably incremented by one
                dirs = glob.glob("DATASET_*", root_dir=pth / state.DAQ_dirname)
                nums = [int(s.split(sep="_")[-1]) for s in dirs]
                if len(nums):
                    dataset_number_being_written = max(nums) + 1
                else:
                    dataset_number_being_written = 0

            STATE_REQUIRES_UPDATE = True
            stream_loc = _create_stream_directory(
                pth, state, dataset_number_being_written=dataset_number_being_written
            )

            streamer.start(stream_loc)
            update_period = 1.0

        old_state = state
        time.sleep(update_period)


if __name__ == "__main__":
    print(sys.argv)
    main(user=sys.argv[1], password=sys.argv[2], pth=sys.argv[3])
