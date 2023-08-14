import os

import librosa
import numpy as np
from scipy.io import wavfile
from glob import glob
from praatio import textgrid


def readtg(tg_path, sample_rate=24000, n_shift=300):
    alignment = textgrid.openTextgrid(tg_path, includeEmptyIntervals=True)
    phones = []
    ends = []
    #for interval in alignment.tierDict["phones"].entryList:
    for interval in alignment.getTier("phones").entries:
        phone = interval.label
        phones.append(phone)
        ends.append(interval.end)
    frame_pos = librosa.time_to_frames(ends, sr=sample_rate, hop_length=n_shift)
    durations = np.diff(frame_pos, prepend=0)
    assert len(durations) == len(phones)
    # merge  "" and sp in the end
    if phones[-1] == "" and len(phones) > 1 and phones[-2] == "sp":
        phones = phones[:-1]
        durations[-2] += durations[-1]
        durations = durations[:-1]
    # replace the last "sp" with "sil" in MFA1.x
    phones[-1] = "sil" if phones[-1] == "sp" else phones[-1]
    # replace the edge "" with "sil", replace the inner "" with "sp"
    new_phones = []
    for i, phn in enumerate(phones):
        if phn == "":
            if i in {0, len(phones) - 1}:
                new_phones.append("sil")
            else:
                new_phones.append("sp")
        else:
            new_phones.append(phn)
    phones = new_phones
    results = ""
    for (p, d) in zip(phones, durations):
        results += p + " " + str(d) + " "
    return phones


def prepare_align(config):
    #import ipdb
    #ipdb.set_trace()
    wav_dir = config["path"]["wav_path"]
    tg_dir = config["path"]["phone_path"]
    lexicon_path = config["path"]["lexicon_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    speaker = "term_mini"
    tg_files = glob(tg_dir + "/*TextGrid")
    print(len(tg_files))
    for tg_file in tg_files:
        prefix = tg_file.split("/")[-1].split(".")[0]
        if not os.path.exists(os.path.join(wav_dir, prefix + ".wav")):
            continue
        text = readtg(tg_file)
        wav_path = os.path.join(wav_dir, os.path.join(wav_dir, prefix + ".wav"))
        if os.path.exists(wav_path):
            os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
            #import ipdb
            #ipdb.set_trace()
            wav, _ = librosa.load(wav_path, sr = sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir, speaker, "{}.wav".format(prefix)),
                sampling_rate,
                wav.astype(np.int16),
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(prefix)),
                "w",
            ) as f1:
                f1.write(" ".join(text))
