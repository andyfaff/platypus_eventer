from multiprocessing import Process, Queue, Event
from functools import partial
import threading
import time
import gzip
import numpy as np
import struct

# import RPi.GPIO as gpio


T0_PIN = 10


def writer(pth, queue):
    with gzip.GzipFile(pth / "EOS.gz", mode="wb", compresslevel=2) as fi:
        while True:
            item = queue.get()
            if item is None:
                fi.flush()
                # multiprocessing.Queue has no task_done
                # queue.task_done()
                print("closing writer")
                break
            fi.write(item)
            # multiprocessing.Queue has no task_done
            # queue.task_done()


nan = np.float16(np.nan).tobytes()


def T0_streamer(frame, frame_event, queue, shutdown_event):
    def _callback(queue, channel):
        while True:
            time.sleep(0.04166)
            t = time.time_ns()
            with frame.get_lock():
                frame.value += 1
                # print(f"frame-{frame.value}")
                b = struct.pack(">LQ2s", frame.value, t, nan)
            queue.put(b)
            frame_event.set()
            if shutdown_event.is_set():
                break

    callback = partial(_callback, queue)

    thread = threading.Thread(target=callback, args=(1,))
    thread.start()

    # gpio.setmode(gpio.BOARD)
    # gpio.setup(T0_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    # gpio.add_event_detect(T0_PIN, gpio.RISING)
    # gpio.add_event_callback(T0_PIN, callback)

    while True:
        shutdown_event.wait(timeout=1.0)
        if shutdown_event.is_set():
            print("T0 streamer stopping")
            thread.join()
            break

    # gpio.cleanup()


def ADC_streamer(frame, frame_event, queue, shutdown_event):
    # TODO ADC pin
    # gpio.setmode(gpio.BOARD)

    while True:
        frame_event.wait(timeout=1)
        if frame_event.is_set():
            # ADC measure
            # print("logged ADC")
            t = time.time_ns()
            with frame.get_lock():
                b = struct.pack(">LQ2s", frame.value, t, np.float16(-1.0).tobytes())
                queue.put(b)
            # this specifies one measurement per frame
            frame_event.clear()

        if shutdown_event.is_set():
            print("ADC streamer stopping")
            break

    # gpio.cleanup()
