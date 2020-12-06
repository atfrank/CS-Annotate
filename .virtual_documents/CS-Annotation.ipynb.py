# test
import rdkit as rd
import deepchem as dc
import numpy as np
import tensorflow as tf
import pandas as pd 


NUMBER_CHEMICAL_SHIFT_TYPE = 19
RETAIN = ['id', 'resid', 'resname', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']
RETAIN_NONAME = ['id', 'resid', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']
import CSRNA *


## Import Module
import pandas as pd
import numpy as np
import io
import pandas as pd
import io
import requests
import deepchem as dc
import tensorflow as tf


################################################################
## Prepare data
################################################################
neighbors = 3
train = get_data(neighbors, training = True)
test = get_data(neighbors, training = False)


# Prepare training set
sd_scale = 0.5
targets = ['sasa', 'astack',  'nastack', 'pair', 'syn_anti']
drop_names = ['id', 'resid', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'astack',  'nastack', 'pair', 'syn_anti', 'pucker', 'class']
tmp_trainX = train.drop(drop_names, axis=1)
tmp_trainy = pd.DataFrame(train[targets].values, dtype = 'float', columns = targets)
trainX = pd.DataFrame(tmp_trainX.values, dtype = 'float')
trainy = tmp_trainy.values
train_mean = [trainy[:, 0].mean()]
train_sd = [sd_scale*np.std(trainy[:, 0])]
tmp = one_hot_encode(pd.DataFrame(train['pucker'].values, columns=['pucker']), ['pucker'])
trainy_mix = np.vstack([ np.where(trainy[:, 0] <= train_mean[0]+train_sd[0], 0, 1),                         
                        trainy[:, 1],
                        trainy[:, 2],
                        trainy[:, 3],
                        trainy[:, 4],
                        tmp['pucker_C2p_endo'].values,
                        tmp['pucker_C3p_endo'].values,
                        tmp['pucker_C2p_exo'].values,
                        tmp['pucker_C3p_exo'].values,
                        tmp['pucker_C1p_exo'].values,
                        tmp['pucker_C4p_exo'].values
                        ]).T
w = np.ones(len(trainy[:, 0]))
train_w = np.vstack([w, w, w, w, w, w, w, w, w, w, w]).T
train_dataset = dc.data.NumpyDataset(trainX, trainy_mix, train_w)

# Prepare test model
retain = ['id', 'resid', 'sasa']
tmp_testX = test.drop(drop_names, axis=1)
tmp_testy = pd.DataFrame(test[targets].values, dtype = 'float', columns = targets)
targets.append('pucker_C2p_endo')
targets.append('pucker_C3p_endo')
targets.append('pucker_C2p_exo')
targets.append('pucker_C3p_exo')
targets.append('pucker_C1p_exo')
targets.append('pucker_C4p_exo')

testX = pd.DataFrame(tmp_testX.values, dtype = 'float')
testy = tmp_testy.values
tmp = one_hot_encode(pd.DataFrame(test['pucker'].values, columns=['pucker']), ['pucker'])
testy_mix = np.vstack([ np.where(testy[:, 0] <= train_mean[0]+train_sd[0], 0, 1),                         
                        testy[:, 1],
                        testy[:, 2],
                        testy[:, 3],
                        testy[:, 4],
                        tmp['pucker_C2p_endo'].values,
                        tmp['pucker_C3p_endo'].values,
                        tmp['pucker_C2p_exo'].values,
                        tmp['pucker_C3p_exo'].values,
                        tmp['pucker_C1p_exo'].values,
                        tmp['pucker_C4p_exo'].values                       
                        ]).T
w = np.ones(len(testy[:, 0]))
test_w = np.vstack([w, w, w, w, w, w, w, w, w, w, w]).T
test_dataset = dc.data.NumpyDataset(np.array(testX), np.array(testy_mix), test_w)
info = pd.DataFrame(test[retain].values, dtype='object', columns = retain)
actuals = pd.DataFrame(testy_mix, columns=targets)



pd.DataFrame({"sasa":trainy_mix[:, 0], 
             "astack":trainy_mix[:, 1], 
             "nastack":trainy_mix[:, 2], 
             "pair":trainy_mix[:, 3], 
             "syn_anti":trainy_mix[:, 4], 
             "pucker_C2p_endo":trainy_mix[:, 5],
             "pucker_C3p_endo":trainy_mix[:, 6],
             "pucker_C2p_exo":trainy_mix[:, 7],
             "pucker_C3p_exo":trainy_mix[:, 8],
             "pucker_C1p_exo":trainy_mix[:, 9],
             "pucker_C4p_exo":trainy_mix[:, 10]}).to_csv("train_target.csv", sep = " ", index = False)

trainX.to_csv("train_features.csv", sep = " ", index = False)

pd.DataFrame({"sasa":testy_mix[:, 0], 
             "astack":testy_mix[:, 1], 
             "nastack":testy_mix[:, 2], 
             "pair":testy_mix[:, 3], 
             "syn_anti":testy_mix[:, 4], 
             "pucker_C2p_endo":testy_mix[:, 5],
             "pucker_C3p_endo":testy_mix[:, 6],
             "pucker_C2p_exo":testy_mix[:, 7],
             "pucker_C3p_exo":testy_mix[:, 8],
             "pucker_C1p_exo":testy_mix[:, 9],
             "pucker_C4p_exo":testy_mix[:, 10]}).to_csv("test_target.csv", sep = " ", index = False)

testX.to_csv("test_features.csv", sep = " ", index = False)


# Scale
transform_scaler = dc.trans.transformers.NormalizationTransformer(transform_X = True, transform_y = False, dataset=train_dataset)
train_dataset_norm = transform_scaler.transform(train_dataset)
test_dataset_norm = transform_scaler.transform(test_dataset)

# Balance Dataset
transform_balancer = dc.trans.transformers.BalancingTransformer(transform_w = True, dataset=train_dataset_norm)

#train_dataset_balanced = balancer.transform(train_dataset_norm)
train_dataset_balanced = balance_transformer(train_dataset_norm)

# confirm not overlap between between training and testing
set(train.id.unique()).intersection(set(test.id.unique()))

n_features = train_dataset_balanced.X.shape[1]
n_tasks = train_dataset_balanced.y.shape[1]


model = dc.models.ProgressiveMultitaskClassifier(n_tasks=n_tasks, n_features=n_features, alpha_init_stddevs=0.04)
model.fit(train_dataset_balanced, nb_epoch=100)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import matplotlib.cm as cm

colors = cm.rainbow(np.linspace(0, 1, len(targets)))
testpred = model.predict(test_dataset_norm)
plt.figure()
lw = 3

# get TPR and FPR
for i,target in enumerate(targets):
  fpr, tpr, thresholds = roc_curve(test_dataset.y[:, i], testpred[:, i, 1].flatten())
  roc_auc = auc(fpr, tpr)
  # Make plot
  plt.plot(fpr, tpr, color=colors[i],lw=lw, label='get_ipython().run_line_magic("s", " (area = %0.2f)' % (target, roc_auc))")

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc=(1.1,0))
plt.show()

predictions = pd.DataFrame(testpred[:, :, 1], columns=["p"+i for i in targets])
predictions = pd.concat([info, actuals, predictions], axis=1)
predictions.to_csv('predictions.txt', sep = ' ')
predictions.tail()


list(tmp.columns)


dc.models.RobustMultitaskClassifier()
