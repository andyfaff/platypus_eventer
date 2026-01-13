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


nan = np.float16(np.nan)


def T0_streamer(frame, frame_event, queue, shutdown_event):
    import RPi.GPIO as gpio

    last_time = [time.time_ns()]

    def _callback(queue, channel):
        t = time.time_ns()
        # debounce
        if t - last_time[0] < 25_000_000:
            return
        last_time[0] = t

        # print("T0")
        with frame.get_lock():
            frame.value += 1
            # print(f"frame-{frame.value}")
            b = struct.pack(_struct, frame.value, t, -1, nan)
        queue.put(b)
        frame_event.set()

    callback = partial(_callback, queue)

    gpio.setmode(gpio.BOARD)
    gpio.setup(T0_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    gpio.add_event_detect(T0_PIN, gpio.RISING)
    gpio.add_event_callback(T0_PIN, callback)

    while True:
        time.sleep(1.0)
        # shutdown_event.wait(1.0)
        if shutdown_event.is_set():
            print("Shutting down T0")
            break
    gpio.cleanup()


def ADC_streamer2(frame, frame_event, queue, shutdown_event, frame_frequency, N):
    # only for testing chopper 4 signal. When you use ADC comment this out again
    import RPi.GPIO as gpio

    def _callback(queue, channel):
        t = time.time_ns()
        # print("T4")
        with frame.get_lock():
            b = struct.pack(_struct, frame.value, t, -4, nan)
        queue.put(b)

    callback = partial(_callback, queue)

    gpio.setmode(gpio.BOARD)
    gpio.setup(T4_PIN, gpio.IN, pull_up_down=gpio.PUD_OFF)
    gpio.add_event_detect(T4_PIN, gpio.RISING, bouncetime=10)
    gpio.add_event_callback(T4_PIN, callback)

    while True:
        time.sleep(1.0)
        # shutdown_event.wait(timeout=1.0)
        if shutdown_event.is_set():
            print("Shutting down T4")
            break

    gpio.cleanup()


def ADC_streamer(frame, frame_event, queue, shutdown_event, frame_frequency, N):
    import spidev

    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1000000

    def read_channel(channel):
        adc = spi.xfer2([1, (8 + channel) << 4, 0])
        data = ((adc[1] & 3) << 8) + adc[2]
        return data

    def convert_volts(data):
        volts = (data * 3.3) / 1023.0
        return volts

    # Samples per frame
    if frame_frequency is None:
        frame_frequency = 25
    period = 1 / frame_frequency
    tspacing = period / N

    while True:
        frame_event.wait(timeout=1)
        # if frame_event.is_set():
        with frame.get_lock():
            f = frame.value

        frame_event.clear()
        for i in range(N):
            t = time.time_ns()
            if i == 0:
                init_time = t

            # ADC measure
            level = read_channel(0)
            v = convert_volts(level)
            # voltage divider 8.05 kOhm, 17.92 kOhm = 0.31 divider
            # measured at 0.306 using 10 V supply and voltmeter
            b = struct.pack(_struct, f, t, 1, np.float16(v))
            queue.put(b)
            if frame_event.is_set() or (t - init_time) * 1e-9 > period - tspacing:
                break
            if i < N - 1:
                time.sleep(tspacing)

        if shutdown_event.is_set():
            print("ADC streamer stopping")
            break
