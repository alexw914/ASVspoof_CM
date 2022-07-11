import os
from random import random
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import colors, pyplot as plt

def reduce_dimension(target_emb, target_dict, output_name, hparams):
    # os.environ['PYTHONHASHSEED'] = str(668)
    # np.random.seed(668)
    X = list(target_emb.values())
    X = TSNE(n_components=2, perplexity=40, init='random').fit_transform(X)
    # X = PCA(n_components=2).fit_transform(X)
    utts = list(target_emb.keys())
    arr = []
    for i in range(len(utts)):
        utt_id = utts[i]
        arr.append(np.append(X[i], [target_dict[utt_id]['bonafide'], utt_id] ))
        # arr.append(np.append(X[i], [gt_data[utt_id]['attack_id'], utt_id]))
    df = pd.DataFrame(arr, columns=['x','y','label','utt_id'])
    df.to_csv(os.path.join(hparams['output_folder'], output_name))


def visualize(df,vis_set="LA"):

    data = []
    for index, row in df.iterrows():
        cur = []
        cur.append(row['x'])
        cur.append(row['y'])
            # if row['label']=="bonafide":
            #     cur.append("bonafide")
            # else:
            #     cur.append("spoof")

        cur.append(row["label"])
        data.append(cur)
    
    data = pd.DataFrame(data, columns=['x','y','label']) 
    new_data = data[data["label"]=="bonafide"][:2000]
    attacks = ["A07","A08","A09"] + ["A1"+str(i) for i in range(10)]
    
    for att in attacks:
        new_data = pd.concat([data[data["label"]==att][:2000],new_data],0)
    # data = pd.concat([data[data["label"]=="spoof"][:2000],data[data["label"]=="bonafide"][:2000]],0)
    # print(data)

    c = ['#CCCC00','#993366']

    if vis_set=="2019LA":
        hue_order = ["bonafide","A07","A08","A09"] + ["A1"+str(i) for i in range(10)]
        ax =sns.scatterplot(x='x', y='y', data=new_data,hue='label',hue_order=hue_order, s=6)
    else:
        hue_order = None
        ax =sns.scatterplot(x='x', y='y', data=data, palette=c,hue='label',hue_order=hue_order, s=6)
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    if vis_set=="2019LA":
        ax.legend(markerscale=0.1, prop={'size':5})
    # plt.tight_layout()
    plt.savefig('./_vis_feat.pdf', dpi=500, bbox_inches="tight")
    plt.show()
    plt.close()


if __name__=="__main__":
    df = pd.read_csv('results/cm/1670/ASVspoof2019-cm_2d.csv')
    visualize(df)