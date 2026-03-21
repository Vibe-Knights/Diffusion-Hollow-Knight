import os
import time
import cv2
import numpy as np
import pandas as pd
from mss import mss
from pynput import keyboard
from pynput.keyboard import Key


FPS = 20
BASE_OUTPUT_DIR = "dataset"

OUTPUT_WIDTH = 128
OUTPUT_HEIGHT = 72

HIGH_RES_WIDTH = 854
HIGH_RES_HEIGHT = 480

SAVE_HIGH_RES = False


TRACKED_KEYS = {
    'a': 'LEFT',
    'd': 'RIGHT',
    'w': 'UP',
    's': 'DOWN',
    'k': 'ATTACK',
    'j': 'HEAL',
    'space': 'JUMP'
}


frame_log = []
key_events = []


def create_unique_dir(base):
    if not os.path.exists(base):
        os.makedirs(base)
        return base

    i = 1
    while True:
        new_dir = f"{base}_{i}"
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            return new_dir
        i += 1


def record_key_event(key_name, event_type):
    key_events.append({
        "timestamp": time.perf_counter(),
        "key": key_name,
        "event": event_type
    })


def on_press(key):
    try:
        k = key.char.lower()
        if k in TRACKED_KEYS:
            record_key_event(TRACKED_KEYS[k], "press")
    except:
        if key == Key.space:
            record_key_event(TRACKED_KEYS["space"], "press")


def on_release(key):
    try:
        k = key.char.lower()
        if k in TRACKED_KEYS:
            record_key_event(TRACKED_KEYS[k], "release")
    except:
        if key == Key.space:
            record_key_event(TRACKED_KEYS["space"], "release")


if __name__ == "__main__":
    OUTPUT_DIR = create_unique_dir(BASE_OUTPUT_DIR)

    FRAME_LOW_DIR = os.path.join(OUTPUT_DIR, "frames_low_res")
    FRAME_HIGH_DIR = os.path.join(OUTPUT_DIR, "frames_high_res")

    os.makedirs(FRAME_LOW_DIR)

    if SAVE_HIGH_RES:
        os.makedirs(FRAME_HIGH_DIR)

    print("Saving dataset to:", OUTPUT_DIR)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    sct = mss()
    monitor = sct.monitors[1]

    frame_time = 1.0 / FPS
    frame_id = 0

    print("Recording")

    try:
        while True:
            start = time.perf_counter()

            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            img_low = cv2.resize(img, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_AREA)

            frame_name = f"{frame_id:07d}.png"
            low_path = os.path.join(FRAME_LOW_DIR, frame_name)

            cv2.imwrite(low_path, img_low, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            if SAVE_HIGH_RES:
                img_up = cv2.resize(img, (HIGH_RES_WIDTH, HIGH_RES_HEIGHT), interpolation=cv2.INTER_AREA)
                up_path = os.path.join(FRAME_HIGH_DIR, frame_name)
                cv2.imwrite(up_path, img_up, [cv2.IMWRITE_PNG_COMPRESSION, 1])

            frame_log.append({
                "frame_id": frame_id,
                "timestamp": start
            })

            frame_id += 1

            elapsed = time.perf_counter() - start
            sleep_time = frame_time - elapsed

            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopped recording")
        listener.stop()

    pd.DataFrame(frame_log).to_csv(os.path.join(OUTPUT_DIR, "frame_times.csv"), index=False)
    pd.DataFrame(key_events).to_csv(os.path.join(OUTPUT_DIR, "key_events.csv"), index=False)

    print(f"Dataset saved in {OUTPUT_DIR}.")
