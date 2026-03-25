import pandas as pd
import os


DATASETS_ROOT = "."
OUTPUT_FILE = "aggregated_dataset.csv"
KEYS = ["LEFT", "RIGHT", "UP", "DOWN", "JUMP", "ATTACK", "HEAL"]


def reconstruct_key_states(frame_times, key_events):
    key_state = {k:0 for k in KEYS}

    events_index = 0
    rows = []

    key_events = key_events.sort_values("timestamp").reset_index(drop=True)

    for _, frame in frame_times.iterrows():
        t = frame["timestamp"]

        while events_index < len(key_events) and key_events.loc[events_index, "timestamp"] <= t:
            event = key_events.loc[events_index]
            key = event["key"]

            if key in key_state:
                if event["event"] == "press":
                    key_state[key] = 1
                else:
                    key_state[key] = 0

            events_index += 1
        rows.append(key_state.copy())

    return pd.DataFrame(rows)


def process_dataset(dataset_path, dataset_id):
    frame_times = pd.read_csv(os.path.join(dataset_path, "frame_times.csv"))
    key_events = pd.read_csv(os.path.join(dataset_path, "key_events.csv"))

    key_states = reconstruct_key_states(frame_times, key_events)
    frame_times = frame_times.reset_index(drop=True)

    result = pd.concat([frame_times, key_states], axis=1)
    result["dataset_id"] = dataset_id

    result["frame_low_res_path"] = result["frame_id"].apply(
        lambda x: os.path.join(dataset_path, "frames_low_res", f"{int(x):07d}.png")
    )

    return result




if __name__ == "__main__":
    datasets = [d for d in os.listdir(DATASETS_ROOT) if d.startswith("dataset")]
    all_data = []

    for i, dataset in enumerate(sorted(datasets)):
        dataset_path = os.path.join(DATASETS_ROOT, dataset)
        print("Processing", dataset)
        df = process_dataset(dataset_path, i)
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Recorded {len(final_df)} frames")
    print("Saved:", OUTPUT_FILE)
