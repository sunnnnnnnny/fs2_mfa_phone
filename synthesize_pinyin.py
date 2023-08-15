import re
import os
import argparse
from string import punctuation

import torch
import yaml
import cn2an
import librosa
import numpy as np
import audio as Audio
from scipy.io import wavfile
from torch.utils.data import DataLoader
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "AISHELL3"
preprocess_config= "config/{}/preprocess.yaml".format(dataset)
model_config = "config/{}/model.yaml".format(dataset)
train_config = "config/{}/train.yaml".format(dataset)
preprocess_config = yaml.load(open(preprocess_config, "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open(model_config, "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open(train_config, "r"), Loader=yaml.FullLoader)
configs = (preprocess_config, model_config, train_config)

pitch_control = 1.0
energy_control = 1.0
duration_control = 1.0
control_values = pitch_control, energy_control, duration_control


lexicon_path = preprocess_config["path"]["lexicon_path_set"]
lexicon_tmp = set()
with open(lexicon_path, "r") as log:
    lines = log.readlines()
    for line in lines:
        phones = line.strip().split("\t")[-1]
        phones_split = phones.split()
        for phone in phones_split:
            lexicon_tmp.add(phone)
lexicon = list(lexicon_tmp)
lexicon.append("sil")
lexicon.append("sp")
lexicon.sort()


def phone_text_to_sequence(text):
    #import ipdb
    #ipdb.set_trace()
    text_split = text.strip().split()
    text_ids = []
    for phone in text_split:
        if phone not in lexicon:
            continue
        text_ids.append(lexicon.index(phone)+1)
    return text_ids


def read_lexicon(lex_path):
    #import ipdb
    #ipdb.set_trace()
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            #temp = re.split(r"\s+", line.strip("\n"))
            temp = line.strip().split(" ", 1)
            word = temp[0]
            phones = temp[1:]
            lexicon[word] = phones            
            #if word.lower() not in lexicon:
            #    lexicon[word.lower()] = phones
    return lexicon
def normalize_float(matchobj):
    float2char = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八",
                  "9": "九"}
    matchobj = matchobj.group(0)
    sub_int = matchobj.split(".")[0]
    transform_int = cn2an.transform(sub_int, "an2cn")
    transform_float = ""
    sub_float = matchobj.split(".")[1]
    ct = 0
    sub_float_tmp = int(sub_float)
    if sub_float_tmp == 0:
        return transform_int
    for idx in range(len(sub_float)-1, -1, -1):
        if sub_float[idx] == '0':
            ct = ct + 1
        else:
            break
    if ct > 0:
        sub_float = sub_float[0:len(sub_float)-ct]
    for str in sub_float:
        transform_float = transform_float + float2char[str]

    return transform_int + "点" + transform_float

def normalize_date(matchobj):
    float2char = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八",
                  "9": "九"}
    matchobj = matchobj.group(0)
    matchobj_split = matchobj.split("-")
    date_res = ""
    for str in matchobj_split[0]:
        date_res = date_res + float2char[str]
    date_res = date_res + "年"
    date_res = date_res + cn2an.transform(matchobj_split[1], "an2cn") + "月"
    date_res = date_res + cn2an.transform(matchobj_split[2], "an2cn") + "日"
    return date_res


def normalize_seq(matchobj):
    float2char = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八",
                  "9": "九"}
    res = ""
    for str in matchobj[0]:
        res = res + float2char[str]
    return res
def normalize_phone_number(matchobj):
    phone_number = matchobj[0]
    float2char = {"0": "零", "1": "幺", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八",
                  "9": "九"}
    res = ""
    for str in matchobj[0]:
        if str not in float2char:
            continue
        res = res + float2char[str]
    return res




def process_normal_pinyin(word_seg_list):
    char_list = "".join(word_seg_list)
    print(char_list)
    res = []
    word2pinyin = {"我行":"wo3 hang2"}
    for idx, word in enumerate(word_seg_list):
        import ipdb
        ipdb.set_trace()
        if word in word2pinyin:
            pinyin = word2pinyin[word]
            pinyin_split = pinyin.split()
            word_seg_list[idx] = pinyin

        else:
            word_seg_list[idx] = " ".join(word)
    word_seg_list = " ".join(word_seg_list).split()

    print(word_seg_list)

def normalize_item(text): 
    float2char = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四", "5": "五", "6": "六", "7": "七", "8": "八",
                  "9": "九"}
    res = ""
    for idx, item in enumerate(text):
        if item in float2char:
            res = res + float2char[item]
        else:
            res = res + item
    return res
    

def text_normalizetion(text):
    text = re.sub("\d{3}-\d{8}", normalize_phone_number, text)
    text = re.sub("\d+(\.\d+)+", normalize_float, text)
    text = re.sub("\d{4}-\d{1,}-\d{1,}", normalize_date, text)
    text = re.sub("\d{4,}", normalize_seq, text)
    text = normalize_item(text)
    return text

def preprocess_mandarin(text, preprocess_config,model_g2p):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    text = text_normalizetion(text)
    phones = []
    if model_g2p is None:
        pinyins = [
            p[0]
            for p in pinyin(
                text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
            )
        ]
    else:
        pinyins = model_g2p(text)[0]
    print("Pinyin Sequence：{}".format(pinyins))
    piny2phone_list = []
    punc = ",.!?，。！？"
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
            piny2phone_list.append("{}-{}".format(p, lexicon[p]))
        else:
            phones.append("sp")
            piny2phone_list.append("{}-{}".format(p, "sp"))
    phones = " ".join(phones) + " sil"
    print("Raw Text Sequence: {}".format(text))
    piny2phone_cat = " *|* ".join(piny2phone_list) 
    print("piny2phone_cat Sequence: {}".format(piny2phone_cat))
    print("Phoneme Sequence: {}".format(phones))
    #import ipdb
    #ipdb.set_trace()
    sequence = phone_text_to_sequence(phones)

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
def synthesize_sub(model, configs, vocoder, batchs, control_values, target_path):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    for batch in batchs:
        print(" ".join(batch[1]))
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            _, wave_path = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                target_path,
            )
        return  _, wave_path
def get_stft_func():
    stft_func = Audio.stft.TacotronSTFT(
        preprocess_config["preprocessing"]["stft"]["filter_length"],
        preprocess_config["preprocessing"]["stft"]["hop_length"],
        preprocess_config["preprocessing"]["stft"]["win_length"],
        preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        preprocess_config["preprocessing"]["mel"]["mel_fmax"],)
    return stft_func
def get_ref_mel(ref_path):
    ref_dir = os.path.dirname(ref_path)
    wav_name = os.path.split(ref_path)[-1]
    ref_sampling_rate = 22050
    wav, _ = librosa.load(ref_path, ref_sampling_rate)
    wav = wav / max(abs(wav)) * 32768.0
    # if not os.path.exists(os.path.join(out_dir, speaker, wav_name)):
    wavfile.write(os.path.join(ref_dir, wav_name.replace(".wav","_22k.wav")), ref_sampling_rate, wav.astype(np.int16), )
    ref_wav_path = os.path.join(ref_dir, wav_name.replace(".wav","_22k.wav"))
    wav, _ = librosa.load(ref_wav_path,sr = ref_sampling_rate)
    stft_func = get_stft_func()
    ref_mel_spec, _ = Audio.tools.get_mel_from_wav(wav, stft_func)
    return ref_mel_spec.T

def text_to_speech(text, ref_wav_path, models):
    mode="single"
    dataset = "AISHELL3"
    speaker_id=0
    ref_mel_spect = get_ref_mel(ref_wav_path)

    if mode == "single":
        ids = raw_texts = [text[:100]]
        speakers = np.array([speaker_id])
        texts = preprocess_mandarin(text, preprocess_config, models["g2p"])
        texts = np.array([texts])
        text_lens = np.array([len(texts[0])])
        ref_mel = np.array([ref_mel_spect])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens),ref_mel_spect)]

    _, wav_path = synthesize_sub(models["model"], configs, models["vocoder"],batchs, control_values, ref_wav_path.replace(".wav","_ref_res.wav"))
    return wav_path
def main(text, ref_wav_path):
    import torch
    from utils.model import get_model, get_vocoder
    # from g2pW.get_convert import conv
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    restore_step=75000
    model = get_model(restore_step, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)
    models = {}
    models["model"] = model
    models["vocoder"] = vocoder
    models["g2p"] = None
    import time
    t0 = time.time()
    wav_path = text_to_speech(text, ref_wav_path, models)
    print(time.time()-t0)
    print(wav_path)


if __name__ == "__main__":
    import sys
    text = "你这里有一笔贷款还没有还,一共222.24元"
    ref_wav_path = sys.argv[1]
    main(text,ref_wav_path)
