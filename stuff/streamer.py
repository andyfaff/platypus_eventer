from multiprocessing import Process, Queue, Event
from functools import partial
import time
import gzip
import numpy as np
import struct
import RPi.GPIO as gpio


T0_PIN = 10


def writer(pth, queue):
    with gzip.GzipFile(pth / "EOS.gz", mode="wb", compresslevel=2) as fi:
        while True:
            item = queue.get()
            if item is None:
                fi.flush()
                queue.task_done()
                break
            fi.write(item)
            queue.task_done()


nan = np.float16(np.nan).tobytes()


def T0_streamer(frame, queue, frame_event, shutdown_event):
    def _callback(queue, channel):
        t = time.time_ns()
        with frame.get_lock():
            frame.value += 1
            b = struct.pack(">LQ2s", frame.value, t, nan)
        queue.put(b)
        frame_event.set()

    callback = partial(_callback, queue)
    gpio.setmode(gpio.BOARD)
    gpio.setup(T0_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    gpio.add_event_detect(T0_PIN, gpio.RISING)
    gpio.add_event_callback(T0_PIN, callback)

    while True:
        shutdown_event.wait(timeout=1.0)
        if shutdown_event.is_set():
            break

    gpio.cleanup()


def ADC_streamer(frame, queue, frame_event, shutdown_event):
    # TODO ADC pin
    gpio.setmode(gpio.BOARD)

    while True:
        frame_event.wait(timeout=1)
        if frame_event.is_set():
            # ADC measure

            t = time.time_ns()
            with frame.get_lock():
                b = struct.pack(">LQ2s", frame.value, t, -1.0)
                queue.put(b)
            # this specifies one measurement per frame
            frame_event.clear()

        if shutdown_event.is_set():
            break

    gpio.cleanup()
