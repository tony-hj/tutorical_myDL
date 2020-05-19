import pandas  as pd
import os

root = 'dataset'

labels = pd.read_csv(os.path.join(root,'annotation.csv'))

name2id = {[i+'.jpg' for i in list(labels['FileID'])][i]:list(labels['SpeciesID'])[i] for i in range(labels.shape[0])} # 文件名没有后缀

species = pd.read_csv(os.path.join(root,'species.csv'))
id2cls = {list(species.ID)[i]:list(species.ScientificName)[i] for i in range(species.shape[0])}
name2cls = {i:id2cls[name2id[i]] for i in name2id}

data = pd.read_csv(os.path.join(root,'training.csv'))
data = data.sample(frac=1)

paths2train = [i+'.jpg' for i in list(data['FileID'])]
labels2train = list(data['SpeciesID'])
paths2test = [i+'.jpg' for i in list(pd.read_csv(os.path.join(root,'test.csv'))['FileID'])]

print('over')