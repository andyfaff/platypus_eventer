from functools import partial
import threading
import time
import gzip
import numpy as np
import struct

T0_PIN = 11
T4_PIN = 13

# frame | time               | channel       | voltage
# ------|--------------------|---------------|--------
# long  | unsigned long long | unsigned char | float16
#
# time is in ns since epoch
# channel = -1  TO
# channel = -4  subsidiary chopper (for checking if frame offset is correct)
# channel >  0  voltage channel
# voltage measured on specific channel
_struct = "<lQbe"
_struct_sz = struct.calcsize(_struct)

def T0_streamer(frame, frame_event, queue, shutdown_event):
    last_time = [time.time_ns()]
    while True:
        time.sleep(1.0)
        # shutdown_event.wait(1.0)
        if shutdown_event.is_set():
            print("Shutting down T0")
            break


def ADC_streamer(frame, frame_event, queue, shutdown_event, frame_frequency, N):
    while True:
        time.sleep(1.0)
        if shutdown_event.is_set():
            print("ADC streamer stopping")
            break
