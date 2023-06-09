import torch
from torch import LongTensor

import commons
from models import SynthesizerTrn
from text import text_to_sequence
import utils

device = "cuda:0" if torch.cuda.is_available() else "cpu"

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(
        text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def load_model(model_path, config_path):
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2+1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    net_g.eval()
    utils.load_checkpoint(model_path, net_g, None)
    return net_g, hps


def inference(net_g, hps, language, text: str, spk, noise_scale, noise_scale_w, length):
    mark = language_marks[language]
    text = f"{mark}{text}{mark}"
    speaker_ids = hps.speakers
    speaker_id = speaker_ids[spk]
    stn_tst = get_text(text, hps, False)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        sid = LongTensor([speaker_id]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=1.0 / length)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    return audio
