import pandas as pd
import numpy as np

def parse_wave_string(s):
    """
    Convert string like:
    '[-1, -6, -7, ...]'
    into numpy array safely
    """
    try:
        s = s.strip()

        # remove brackets
        if s.startswith("["):
            s = s[1:]
        if s.endswith("]"):
            s = s[:-1]

        # split + convert
        values = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
        return np.array(values, dtype=np.float32)

    except Exception as e:
        return None


def load_waveforms(tsv_path):
    df = pd.read_csv(tsv_path, sep="\t")

    print("Total rows in waveform file:", len(df))

    wave_dict = {}

    # group by match_id (each = 12 leads)
    grouped = df.groupby("match_id")

    for match_id, group in grouped:
        leads = []

        for _, row in group.iterrows():
            wave = parse_wave_string(row["Datas"])

            if wave is None:
                continue

            leads.append(wave)

        # keep only if valid
        if len(leads) >= 1:
            # pad to same length
            min_len = min(len(w) for w in leads)
            leads = [w[:min_len] for w in leads]

            wave_dict[str(match_id)] = np.stack(leads)  # shape (L, T)

    print("Loaded waveform samples:", len(wave_dict))

    return wave_dict