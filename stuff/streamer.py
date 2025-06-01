from multiprocessing import Process, Queue, Event
import threading
from functools import partial
import time
import gzip
import numpy as np
import struct
from pathlib import Path
import RPi.GPIO as gpio


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


def streamer(producer_queue, consumer_queue):
    frame = 0

    def callback(producer_queue, ev, channel):
        t = time.time_ns()
        frame += 1
        b = struct.pack(">LQ2s", frame, t, nan)
        producer_queue.put(b)

    while True:
        pass
