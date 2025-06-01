from multiprocessing import Process, Queue, Value, Event
import ctypes
import time
from pathlib import Path
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
    def acquiring(self):
        return self.dct["DAQ"] == "Started"

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
        self.currently_streaming = False
        self.frame = Value(ctypes.c_uint32)

    def stop(self):
        self.currently_streaming = False
        self.shutdown_event.set()
        self.p_t0_streamer.join()
        self.p_t0_streamer.close()
        self.p_ADC_streamer.join()
        self.p_ADC_streamer.close()

        self.queue.put(None)
        self.queue.join()
        self.p_writer.join()
        self.p_writer.close()

    def start(self, stream_loc):
        self.stream_loc = stream_loc
        self.shutdown_event.clear()
        self.frame_event.clear()

        self.frame = Value(ctypes.c_uint32)
        self.queue = Queue()
        self.p_writer = Process(
            target=streamer.writer, args=(stream_loc, self.queue, self.shutdown_event)
        )
        self.p_t0_streamer = Process(
            target=streamer.T0_streamer,
            args=(self.frame, self.queue, self.frame_event, self.shutdown_event),
        )
        self.p_ADC_streamer = Process(
            target=streamer.ADC_streamer,
            args=(self.frame, self.queue, self.frame_event, self.shutdown_event),
        )

        self.p_writer.start()
        self.p_t0_streamer.start()

        self.currently_streaming = True


def _signal_close(streamer, signal, frame):
    streamer.stop()
    sys.exit()


def main(user, password="", pth=None):
    if pth is None:
        pth = Path.cwd()

    currently_streaming = False

    s = Status(user, password=password, url=url)
    old_state = State(s())
    streamer = EventStreamer()

    # shutdown gracefully
    signal_close = partial(_signal_close, streamer)
    signal.signal(signal.SIGINT, signal_close)

    update_period = 2.0

    while True:
        _s = s()
        state = State(_s)

        if streamer.currently_streaming and (
            not state.acquiring
            or state.DAQ_dirname != old_state.DAQ_dirname
            or state.DATASET_number != old_state.DATASET_number
        ):
            # stop streaming
            streamer.stop()
            update_period = 2

        if state.acquiring and not streamer.currently_streaming:
            # time to start new directories and start streaming
            if (
                state.DAQ_dirname != old_state.DAQ_dirname
                or state.DATASET_number != old_state.DATASET_number
            ):
                stream_loc = pth / state.DAQ_dirname / f"DATA_{state.DATASET_number}"
                stream_loc.mkdir(parents=True)
                print(f"New sample event file started, {stream_loc=}")
                with open(stream_loc, "w") as f:
                    f.write(_s)

            streamer.start(stream_loc)
            update_period = 10.0

        old_state = state
        time.sleep(update_period)


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1], password=sys.argv[2], pth=sys.argv[3])
