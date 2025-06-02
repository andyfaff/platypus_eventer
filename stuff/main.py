from multiprocessing import Process, Queue, Value, Event
import ctypes
import time
from pathlib import Path
import os
import glob
import signal
import sys
from functools import partial
from status import Status, parse_status
import streamer


url = "http://localhost:60001/admin/textstatus.egi"


class State:
    def __init__(self, response):
        status, units = parse_status(response)
        self.dct = {}
        self.dct.update(status)

    @property
    def started(self):
        return self.dct["DAQ"] == "Started"

    @property
    def starting(self):
        return self.dct["DAQ"] == "Starting"

    @property
    def DAQ_dirname(self):
        return self.dct["DAQ_dirname"]

    @property
    def DATASET_number(self):
        return self.dct["DATASET_number"]


class EventStreamer:
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
            new_name = parent / f"DATA_{DATASET_number}"
            self.stream_loc.rename(new_name)
            self.stream_loc = new_name
            with open(self.stream_loc / "final_state.txt", "w") as f:
                f.write(final_state)

    def start(self, stream_loc):
        self.stream_loc = stream_loc
        self.shutdown_event.clear()
        self.frame_event.clear()

        self.frame = Value(ctypes.c_uint32)
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
    streamer.stop()
    sys.exit()


def main(user, password="", pth=None):
    if pth is None:
        pth = Path.cwd()
    else:
        pth = Path(pth)

    s = Status(user, password=password, url=url)
    old_state = State(s())
    streamer = EventStreamer()

    # shutdown gracefully
    signal_close = partial(_signal_close, streamer)
    signal.signal(signal.SIGINT, signal_close)

    update_period = 2.0

    actual_dataset_number = 0
    dataset_number_being_written = 0
    LAST_DAQ_DIRNAME = ""

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
            if LAST_DAQ_DIRNAME != state.DAQ_dirname:
                # totally new dataset
                LAST_DAQ_DIRNAME = state.DAQ_dirname
                dataset_number_being_written = 0
            elif state.started:
                dataset_number_being_written = state.DATASET_number
            elif state.starting:
                # dataset number probably incremented by one
                dirs = glob.glob("DATA_*", root_dir=pth / state.DAQ_dirname)
                nums = [int(s.split(sep="_")[-1]) for s in dirs]
                if len(nums):
                    dataset_number_being_written = max(nums) + 1
                else:
                    dataset_number_being_written = 0

            stream_loc = (
                pth / state.DAQ_dirname / f"DATA_{dataset_number_being_written}"
            )
            stream_loc = stream_loc.resolve()

            os.makedirs(stream_loc, exist_ok=True)
            print(f"New sample event file starting, {stream_loc=}")
            with open(stream_loc / "state.txt", "w") as f:
                f.write(_s)

            streamer.start(stream_loc)
            update_period = 1.0

        old_state = state
        time.sleep(update_period)


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1], password=sys.argv[2], pth=sys.argv[3])
