import os
import random
import json,csv
import glob
from pathlib import Path
from speechbrain.dataio.dataio import read_audio


CM_EVAL2021_FILE_META = 'keys/LA/CM/trial_metadata.txt'
CM_DF_FILE_META = "keys/DF/CM/trial_metadata.txt"
CM_FLAC_DIR = 'ASVspoof2019/LA/ASVspoof2021_%s_eval/flac'
PROCESSED_DATA_DIR = 'cm_meta'

CM_EVAL2021_PROGRESS_FILE = 'cm_eval2021_progress.csv'
CM_EVAL2021_EVAL_FILE = 'cm_eval2021_eval.csv'
CM_DF_PROGRESS_FILE = 'cm_df_progress.csv'
CM_DF_EVAL_FILE = 'cm_df_eval.csv'

BONAFIDE = 'bonafide'
SPOOF = 'spoof'

RANDOM_SEED = 999999
SAMPLERATE = 16000
SPLIT = ['LA','DF']

random.seed(RANDOM_SEED)
Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)

save_files = [[CM_EVAL2021_PROGRESS_FILE,CM_EVAL2021_EVAL_FILE],[CM_DF_PROGRESS_FILE, CM_DF_EVAL_FILE]]
#Check if we already have SpeechBrain format CM protocol
if os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_EVAL2021_PROGRESS_FILE)) and \
    os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_EVAL2021_EVAL_FILE)) and \
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_DF_PROGRESS_FILE)) and \
            os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_DF_EVAL_FILE)):
    print('SpeechBrain format CM protocols exist...')
else:
    print('Start to convert original CM protocols to SpeechBrain format...')
    # Read CM protocols in train/dev/eval set
    for i, file in enumerate([CM_EVAL2021_FILE_META, CM_DF_FILE_META]):
        cm_features_progress = {}
        cm_features_eval = {}
        with open(os.path.join(file), 'r') as f:
            cm_pros = f.readlines()
            print('% has %d data!'%(file, len(cm_pros)))
        for pro in cm_pros:
            pro = pro.strip('\n').split(' ')
            auto_file_name = pro[1]
            bonafide = pro[5]
            phase = pro[7]
            if phase=="progress":
                cm_features_progress[auto_file_name] = {'bonafide': bonafide,}
            cm_features_eval[auto_file_name] = {'bonafide': bonafide,}
        # Read flac files and durations
        cur_flac_files = glob.glob(os.path.join( CM_FLAC_DIR%(SPLIT[i]),'*.flac'),
                                   recursive=True)

        n_miss = 0
        # Read each utt file and get its duration. Update cm features
        for file in cur_flac_files:
            signal = read_audio(file)
            duration = signal.shape[0] / SAMPLERATE
            utt_id = Path(file).stem
            if utt_id in cm_features_progress:
                cm_features_progress[utt_id]['file_path'] = file
                cm_features_progress[utt_id]['duration'] = duration
            if utt_id in cm_features_eval:
                cm_features_eval[utt_id]['file_path'] = file
                cm_features_eval[utt_id]['duration'] = duration

            else: n_miss += 1
        

        print('%d files missed description in protocol file in %s set'%(n_miss, SPLIT[i]))
        with open(os.path.join(PROCESSED_DATA_DIR, save_files[i][0]), 'w') as f:
            csv_writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["ID", "bonafide", "file_path", "duration"])
            for key in cm_features_progress:
                csv_writer.writerow([key, cm_features_progress[key]["bonafide"],cm_features_progress[key]["file_path"], cm_features_progress[key]['duration']])     

        with open(os.path.join(PROCESSED_DATA_DIR, save_files[i][1]), 'w') as f:
            csv_writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["ID", "bonafide", "file_path", "duration"])
            for key in cm_features_eval:
                csv_writer.writerow([key, cm_features_eval[key]["bonafide"],cm_features_eval[key]["file_path"], cm_features_eval[key]['duration']])        
