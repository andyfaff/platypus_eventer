from functools import partial
import threading
import time
import gzip
import numpy as np
import struct

T0_PIN = 11
T4_PIN = 13


def writer(pth, queue):
    with gzip.GzipFile(pth / "EOS.gz", mode="wb", compresslevel=4) as fi:
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
    import RPi.GPIO as gpio

    def _callback(queue, channel):
        t = time.time_ns()
        # print("T0")
        with frame.get_lock():
            frame.value += 1
            # print(f"frame-{frame.value}")
            b = struct.pack(">LQ2s", frame.value, t, nan)
        queue.put(b)
        frame_event.set()

    callback = partial(_callback, queue)

    gpio.setmode(gpio.BOARD)
    gpio.setup(T0_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    gpio.add_event_detect(T0_PIN, gpio.RISING, bouncetime=20)
    gpio.add_event_callback(T0_PIN, callback)

    while True:
        shutdown_event.wait(1.0)
        if shutdown_event.is_set():
            print("Shutting down T0")
            break
    gpio.cleanup()


def ADC_streamer(frame, frame_event, queue, shutdown_event):
    # only for testing chopper 4 signal. When you use ADC comment this out again
    import RPi.GPIO as gpio

    def _callback(queue, channel):
        t = time.time_ns()
        # print("T4")
        with frame.get_lock():
            b = struct.pack(">LQ2s", frame.value, t, np.float16(4).tobytes())
        queue.put(b)

    callback = partial(_callback, queue)

    gpio.setmode(gpio.BOARD)
    gpio.setup(T4_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    gpio.add_event_detect(T4_PIN, gpio.RISING, bouncetime=20)
    gpio.add_event_callback(T4_PIN, callback)

    while True:
        shutdown_event.wait(timeout=1.0)
        if shutdown_event.is_set():
            print("Shutting down T4")
            break

    gpio.cleanup()


# def ADC_streamer(frame, frame_event, queue, shutdown_event):
#     # TODO ADC pin
#     # gpio.setmode(gpio.BOARD)
#
#     while True:
#         frame_event.wait(timeout=1)
#         if frame_event.is_set():
#             # ADC measure
#             # print("logged ADC")
#             t = time.time_ns()
#             with frame.get_lock():
#                 b = struct.pack(">LQ2s", frame.value, t, np.float16(-1.0).tobytes())
#                 queue.put(b)
#             # this specifies one measurement per frame
#             frame_event.clear()
#
#         if shutdown_event.is_set():
#             print("ADC streamer stopping")
#             break
#
#     # gpio.cleanup()
