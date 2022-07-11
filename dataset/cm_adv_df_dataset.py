from cProfile import label
import speechbrain as sb
import numpy as np
import os, torch, sys
import soundfile as sf
from sympy import sequence
import json,librosa
from random import sample
from dataset.speech_process import load_wav
from torchaudio.transforms import Spectrogram

aug_path="/home/alex/Corpora/ASVspoof2019/LA/LADF_AUG"
LABEL_DIR = 'cm_meta'
channel = [('flac','amr[br=10k2,nodtx]', 'amr[br=5k9]', 'amr[br=6k7,nodtx]',
                        'amr[br=7k95,nodtx]', 'amrwb[br=12k65]', 'amrwb[br=15k85]', 'g711[law=a]',
                        'g711[law=u]', 'g722[br=64k]', 'g726[law=a,br=16k]', 'g726[law=a,br=24k]',
                        'g726[law=u,40k]', 'g726[law=u,br=24k]', 'g726[law=u,br=32k]', 'g728',
                        'silk[br=10k,loss=10]', 'silk[br=15k,loss=5]', 'silk[br=15k]',
                        'silk[br=20k,loss=5]', 'silk[br=5k,loss=10]', 'silk[br=5k]', 'amr[br=12k2]',
                        'amr[br=5k9,nodtx]', 'amrwb[br=6k6,nodtx]', 'g722[br=56k]', 'g726[law=a,br=32k]',
                        'g726[law=a,br=40k]','silk[br=15k,loss=10]', 'silk[br=20k]',
                        'silkwb[br=10k,loss=5]', 'amr[br=10k2]', 'amr[br=4k75]', 'amr[br=7k95]',
                        'amrwb[br=15k85,nodtx]', 'amrwb[br=23k05]', 'g726[law=u,br=16k]', 'g729a',
                        'gsmfr', 'silkwb[br=10k,loss=10]', 'silkwb[br=20k]', 'silkwb[br=30k,loss=10]',
                        'amr[br=7k4,nodtx]', 'amrwb[br=6k6]', 'silk[br=10k]', 'silk[br=5k,loss=5]',
                        'silkwb[br=30k,loss=5]', 'amr[br=4k75,nodtx]', 'amr[br=7k4]', 'g722[br=48k]',
                        'silk[br=20k,loss=10]', 'silkwb[br=30k]', 'amr[br=5k15]',
                        'silkwb[br=20k,loss=5]', 'amrwb[br=23k05,nodtx]', 'amrwb[br=12k65,nodtx]',
                        'silkwb[br=20k,loss=10]', 'amr[br=6k7]', 'silkwb[br=10k]', 'silk[br=10k,loss=5]')]
compression = [('flac','aac[16k]', 'aac[32k]', 'aac[8k]', 'mp3[16k]', 'mp3[32k]', 'mp3[8k]')]

def get_adv_dataset(hparams):
    """
    Code here is basically same with code in SpoofSpeechDataset.py
    However, audio will not be load directly.
    A random compression will be made before load by torchaudio
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder_cm = sb.dataio.encoder.CategoricalEncoder()
    # label_encoder_attack = sb.dataio.encoder.CategoricalEncoder()
    label_encoder_domain = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = load_wav(file_path)
        # sig = load_wav_cqt(file_path)
        return sig

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("bonafide")
    @sb.utils.data_pipeline.provides("bonafide", "bonafide_encoded")
    def bonafide_label_pipeline(bonafide):
        yield bonafide
        bonafide_encoded = label_encoder_cm.encode_label_torch(bonafide, True)
        yield bonafide_encoded

    # @sb.utils.data_pipeline.takes("attack")
    # @sb.utils.data_pipeline.provides("attack", "attack_encoded")
    # def attack_label_pipeline(attack):
    #     yield attack
    #     attack_encoded = label_encoder_attack.encode_label_torch(attack, True)
    #     yield attack_encoded

    @sb.utils.data_pipeline.takes("domain")
    @sb.utils.data_pipeline.provides("domain", "domain_encoded")
    def codec_label_pipeline(domain):
        yield domain
        domain_encoded = label_encoder_domain.encode_label_torch(domain, True)
        yield domain_encoded


    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}

    for dataset in ["train"]:
        # print(hparams[f"{dataset}_annotation"])
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path = 'cm_meta/cm_adv_df.csv',
            dynamic_items=[audio_pipeline,bonafide_label_pipeline,codec_label_pipeline],
            output_keys=["id", "sig","bonafide", "bonafide_encoded","domain","domain_encoded"],
        )

    label_encoder_cm.load_or_create(
        path = os.path.join(hparams["save_folder"], "label_encoder_cm.txt"),
        sequence_input = False,
        from_iterables = [("spoof","bonafide")]
    )

    # label_encoder_attack.load_or_create(
    #     path = os.path.join(hparams["save_folder"], "attack_encoder_cm.txt"),
    #     sequence_input = False,
    #     from_iterables = [("-","A01","A02","A03","A04","A05","A06")]
    # )
    
    label_encoder_domain.load_or_create(
        path = os.path.join(hparams["save_folder"], "domain_encoder_cm.txt"),
        sequence_input=False,
        from_didatasets=[datasets["train"]],
        output_key="domain"
    )
                              
    return datasets