import os
import random
import json, csv
import glob
from pathlib import Path
from speechbrain.dataio.dataio import read_audio


CM_PROTO_DIR = 'ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols'
CM_TRAIN_FILE = 'ASVspoof2019.LA.cm.train.trn.txt'
CM_DEV_FILE = 'ASVspoof2019.LA.cm.dev.trl.txt'
CM_EVAL_FILE = 'ASVspoof2019.LA.cm.eval.trl.txt'
CM_FLAC_DIR = 'ASVspoof2019/LA/ASVspoof2019_LA_%s/flac'
PROCESSED_DATA_DIR = 'cm_meta'

CM_SB_TRAIN_FILE = 'cm_train.csv'
CM_SB_DEV_FILE = 'cm_dev.csv'
CM_SB_EVAL_FILE = 'cm_eval.csv'

# Statistics
RANDOM_SEED = 97271
SAMPLERATE = 16000
SPLIT = ['train', 'dev', 'eval']
random.seed(RANDOM_SEED)
Path(PROCESSED_DATA_DIR).mkdir(exist_ok=True)

#Check if we already have SpeechBrain format CM protocol
if os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_TRAIN_FILE)) and \
        os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_DEV_FILE)) and \
            os.path.exists(os.path.join(PROCESSED_DATA_DIR, CM_SB_EVAL_FILE)):
    print('SpeechBrain format CM protocols exist...')
else:
    print('Start to convert original CM protocols to SpeechBrain format...')
    # Read CM protocols in train/dev/eval set
    save_files = [CM_SB_TRAIN_FILE, CM_SB_DEV_FILE, CM_SB_EVAL_FILE]
    for i, file in enumerate([CM_TRAIN_FILE, CM_DEV_FILE, CM_EVAL_FILE]):
        cm_features = {}
        with open(os.path.join(CM_PROTO_DIR, file), 'r') as f:
            cm_pros = f.readlines()
            print('% has %d data!'%(file, len(cm_pros)))
        for pro in cm_pros:
            pro = pro.strip('\n').split(' ')
            speaker_id = pro[0]
            auto_file_name = pro[1]
            spoof_id = pro[3]
            bonafide = pro[4]
            cm_features[auto_file_name] = {
                'speaker_id': speaker_id,
                'attack': spoof_id,
                'bonafide': bonafide,
            }
        # Read flac files and durations
        cur_flac_files = glob.glob(os.path.join( CM_FLAC_DIR%(SPLIT[i]),'*.flac'),
                                   recursive=True)

        n_miss = 0
        # Read each utt file and get its duration. Update cm features

        for file in cur_flac_files:
            signal = read_audio(file)
            duration = signal.shape[0] / SAMPLERATE
            utt_id = Path(file).stem
            if utt_id in cm_features:
                cm_features[utt_id]['file_path'] = file
                cm_features[utt_id]['duration'] = duration
            else: n_miss += 1
        

        print('%d files missed description in protocol file in %s set'%(n_miss, SPLIT[i]))
        # Save updated cm features into json
        with open(os.path.join(PROCESSED_DATA_DIR, save_files[i]), 'w') as f:
            csv_writer = csv.writer(
                f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
            )
            csv_writer.writerow(["ID","speaker_id","attack", "bonafide","file_path", "duration"])
            for key in cm_features:
                csv_writer.writerow([key, cm_features[key]["speaker_id"], cm_features[key]["attack"],cm_features[key]["bonafide"], cm_features[key]['file_path'], cm_features[key]['duration']])




