import speechbrain as sb
import os,json
from dataset.speech_process import load_wav

LABEL_DIR = 'cm_meta'
def get_eval_dataset(hparams):

    label_encoder_cm = sb.dataio.encoder.CategoricalEncoder()

    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(file_path):
        sig = load_wav(file_path, num_frames=748)
        # sig = load_wav_cqt(file_path)
        return sig
    
    @sb.utils.data_pipeline.takes("bonafide")
    @sb.utils.data_pipeline.provides("bonafide", "bonafide_encoded")
    def bonafide_label_pipeline(bonafide):
        yield bonafide
        bonafide_encoded = label_encoder_cm.encode_label_torch(bonafide, True)
        yield bonafide_encoded


    data_dir = 'cm_meta'
    CM_EVAL2021_PROGRESS_FILE = 'cm_eval2021_progress.csv'
    CM_EVAL2021_EVAL_FILE = 'cm_eval2021_eval.csv'
    CM_DF_PROGRESS_FILE = 'cm_df_progress.csv'
    CM_DF_EVAL_FILE = 'cm_df_eval.csv'

    datasets = {}

    datasets['eval_2019'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["eval_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline,bonafide_label_pipeline],
            output_keys=["id", "sig","bonafide", "bonafide_encoded",],
    )

    datasets['eval_2019_progress'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=hparams["dev_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline,bonafide_label_pipeline],
            output_keys=["id", "sig","bonafide", "bonafide_encoded",],
    )

    datasets['eval_2021_progress'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path= os.path.join(data_dir, CM_EVAL2021_PROGRESS_FILE),
        dynamic_items=[audio_pipeline,bonafide_label_pipeline],
        output_keys=["id", "sig", "bonafide", "bonafide_encoded",],
    )

    datasets['eval_2021'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path= os.path.join(data_dir, CM_EVAL2021_EVAL_FILE),
        dynamic_items=[audio_pipeline,bonafide_label_pipeline],
        output_keys=["id", "sig", "bonafide", "bonafide_encoded",],
    )

    datasets['df_progress'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path= os.path.join(data_dir, CM_DF_PROGRESS_FILE),
        dynamic_items=[audio_pipeline,bonafide_label_pipeline],
        output_keys=["id", "sig", "bonafide", "bonafide_encoded",],
    )


    datasets['df_eval'] = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path= os.path.join(data_dir, CM_DF_EVAL_FILE),
        dynamic_items=[audio_pipeline,bonafide_label_pipeline],
        output_keys=["id", "sig", "bonafide", "bonafide_encoded",],
    )


    label_encoder_cm.load_or_create(
        path = os.path.join(hparams["save_folder"], "label_encoder_cm.txt"),
        sequence_input = False,
        from_iterables = [("spoof","bonafide")]
    )
    
    return datasets