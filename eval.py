import pandas as pd
import torch, json, os, sys,random
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from models.cm_2021 import CM
from dataset.cm_eval_dataset import get_eval_dataset
import logging
from collections import defaultdict
from pathlib import Path
from utils.save_dicts import save_scores
from utils.visualize import visualize, reduce_dimension
from utils.eval_asvspoof import la19_eval,la_eval,df_eval
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')

DATA_DIR = 'cm_meta'
OUTPUT_DIR = 'predictions'
Path(OUTPUT_DIR).mkdir(exist_ok=True)

if __name__ == "__main__":

    VISUALIZE = True 
    SCORE = True

    logger = logging.getLogger(__name__)
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams_file = os.path.join(hparams['output_folder'], 'hyperparams.yaml')
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams['batch_size'] = 128
    hparams['dataloader_options']['batch_size'] = 128
    hparams['dataloader_options']['shuffle'] = False


    datasets = get_eval_dataset(hparams)

    encoder = CM(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    save_file_dict = {
        "test_cm_emb_file": os.path.join(hparams['output_folder'],hparams["eval_option"]+'-eval_cm_emb.pt'),
        "test_cm_scores_file":os.path.join(hparams['output_folder'],hparams["eval_option"]+'-eval_cm_scores.pt'),
        "test_cm_emb_progress_file": os.path.join(hparams['output_folder'],hparams["eval_option"]+'-progress-test_cm_emb.pt'),
        "test_cm_scores_progress_file":os.path.join(hparams['output_folder'],hparams["eval_option"]+'-progress-test_cm_scores.pt'),
    }    
    
    if not os.path.exists(save_file_dict["test_cm_emb_file"]):
        save_scores(datasets, encoder, hparams, save_file_dict, hparams["eval_option"])

    test_cm_scores_dict = torch.load(save_file_dict["test_cm_scores_file"])
    test_cm_emb_dict = torch.load(save_file_dict["test_cm_emb_file"])
    test_cm_scores_progress_dict = torch.load(save_file_dict["test_cm_scores_progress_file"])
    test_cm_emb_progress_dict = torch.load(save_file_dict["test_cm_emb_progress_file"])

    test_data_dict = defaultdict(dict)
    with open('keys/DF/CM/trial_metadata.txt','r') as f:
        test_data = f.readlines()
    
    for utt in test_data:
        _, utt_id, _, _, _, bonafide, _, _  = utt.split()
        test_data_dict[utt_id]['bonafide'] = bonafide
    
    for utt in test_data:
        _, utt_id, _, _, _, bonafide, _, _  = utt.split()
        test_data_dict[utt_id]['bonafide'] = bonafide

    if hparams["eval_option"] == "2019LA":
        key_file = 'keys/2019LA/ASV/ASVspoof2019_LA_eval_asv_scores.txt'
    elif hparams["eval_option"] == "2021LA":
        key_file = 'keys/LA/CM/trial_metadata.txt'
    elif hparams["eval_option"] == "2021DF":
        key_file = 'keys/DF/CM/trial_metadata.txt'
    else:
        raise Exception

    with open(key_file,'r') as f:
        test_data = f.readlines()
    if hparams["eval_option"] == "2019LA":
        for utt in test_data:
            _, utt_id, _, bonafide, target, _ = utt.split()
            test_data_dict[utt_id]['bonafide'] = bonafide
            test_data_dict[utt_id]['traget'] = target
    elif hparams["eval_option"] == "2021LA" or hparams["eval_option"] == "2021DF":
        for utt in test_data:
            _, utt_id, _, _, _, bonafide, _, _  = utt.split()
            test_data_dict[utt_id]['bonafide'] = bonafide
    

    if SCORE:
        save_score_file = os.path.join('', OUTPUT_DIR, str(hparams["seed"]) + hparams["eval_option"])
        
        if hparams["eval_option"] == "2019LA":
            with open(save_score_file, "w") as f:
                for key,value in test_cm_scores_dict.items():
                    f.write('%s %s %s %s\n' % (key, value, "bonafide" if test_data_dict[key]['bonafide']=="bonafide" else "spoof", test_data_dict[key]['bonafide']))
            cm_eer, min_tDCF = la19_eval(save_score_file)
            print("CM EER: {:.2f}".format(cm_eer), "min_tDCF: {:.4f}".format(min_tDCF)) 
        elif hparams["eval_option"] == "2021LA" or hparams["eval_option"] == "2021DF": 
            with open(save_score_file, "w") as f:
                for key,value in test_cm_scores_dict.items():
                    f.write('%s %s\n' % (key, value))
                    # f.write('%s %s %s\n' % (key, value, "bonafide" if test_data_dict[key]['bonafide']=="bonafide" else "spoof")) #This option type is for matlab det plot.
            if hparams["eval_option"] == "2021LA": 
                cm_eer, min_tDCF = la_eval(save_score_file, phase="eval")
                print("CM EER: {:.2f}".format(cm_eer), "min_tDCF: {:.4f}".format(min_tDCF))
            if hparams["eval_option"] == "2021DF": 
                cm_eer = df_eval(save_score_file, phase="eval")
                print("CM EER: {:.2f}".format(cm_eer))

    if VISUALIZE:
        # print(os.path.join(hparams['output_folder'], 'cm_2d.csv'))
        csv_2d_file = "cm_"+ hparams["eval_option"] + ".csv" 
        if os.path.exists(os.path.join(hparams['output_folder'], csv_2d_file)):
            df = pd.read_csv(os.path.join(hparams['output_folder'], csv_2d_file))
            visualize(df,hparams["eval_option"])
        else:
            if hparams["eval_option"]=="2019LA":
                for idx, utt_id in enumerate(test_cm_emb_dict):
                    test_cm_emb_dict[utt_id] = torch.squeeze(test_cm_emb_dict[utt_id],0)
                    test_cm_emb_dict[utt_id] = torch.squeeze(test_cm_emb_dict[utt_id],0)
                    test_cm_emb_dict[utt_id] = test_cm_emb_dict[utt_id].cpu().numpy()
                reduce_dimension(test_cm_emb_dict, test_data_dict, csv_2d_file, hparams)
            else:
                for idx, utt_id in enumerate(test_cm_emb_progress_dict):
                    test_cm_emb_progress_dict[utt_id] = torch.squeeze(test_cm_emb_progress_dict[utt_id],0)
                    test_cm_emb_progress_dict[utt_id] = torch.squeeze(test_cm_emb_progress_dict[utt_id],0)
                    test_cm_emb_progress_dict[utt_id] = test_cm_emb_progress_dict[utt_id].cpu().numpy()
                reduce_dimension(test_cm_emb_progress_dict, test_data_dict, csv_2d_file, hparams)
            # process all data
            df = pd.read_csv(os.path.join(hparams["output_folder"], csv_2d_file))
            visualize(df,hparams["eval_option"])
            
    del test_cm_emb_dict, test_cm_scores_dict, test_data_dict


