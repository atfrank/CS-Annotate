# deepchem
import rdkit as rd
import deepchem as dc
import tensorflow as tf

# other
import numpy as np
import pandas as pd 
import io


DIR_PATH = '/Users/aaronfranklab/Documents/Github/CS-Annotate/data/'
RETAIN = ['id', 'resid', 'resname', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']
RETAIN_NONAME = ['id', 'resid', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']
neighbors = 3

"""# Load data and preprocess"""

neighbors = 3
# load train and test data
X_train = pd.read_csv(DIR_PATH+"train_features_"+str(neighbors)+".csv",delim_whitespace=True,header=0)
y_train = pd.read_csv(DIR_PATH+"train_target_"+str(neighbors)+".csv",delim_whitespace=True,header=0)
X_test = pd.read_csv(DIR_PATH+"test_features_"+str(neighbors)+".csv",delim_whitespace=True,header=0)
y_test = pd.read_csv(DIR_PATH+"test_target_"+str(neighbors)+".csv",delim_whitespace=True,header=0)
targets = y_train.columns

# convert to deepchem dataset
w = np.ones(y_train.shape[0]) # number of samples in train
train_w = np.vstack([w, w, w, w, w, w, w, w, w, w, w]).T # weight is 1 
train_dataset = dc.data.NumpyDataset(X_train, y_train, train_w) # use deepchem here, some kind of weight

w = np.ones(y_test.shape[0]) # number of samples in test
test_w = np.vstack([w, w, w, w, w, w, w, w, w, w, w]).T # weight is 1 
test_dataset = dc.data.NumpyDataset(X_test, y_test, test_w) # use deepchem here, some kind of weight

# Scale
transform_scaler = dc.trans.transformers.NormalizationTransformer(transform_X = True, transform_y = False, dataset=train_dataset)
train_dataset_norm = transform_scaler.transform(train_dataset)
test_dataset_norm = transform_scaler.transform(test_dataset)

# Balance Dataset
transform_balancer = dc.trans.transformers.BalancingTransformer(transform_w = True, dataset=train_dataset_norm) # this will work, inconsistent names before
train_dataset_balanced = transform_balancer.transform(train_dataset_norm)

n_features = train_dataset_balanced.X.shape[1]
n_tasks = train_dataset_balanced.y.shape[1]


"""# Train, Save and Test Model"""

targets

# save model
model = dc.models.ProgressiveMultitaskClassifier(n_tasks=n_tasks,n_features=n_features,layer_sizes=[100],alpha_init_stddevs=0.04,learning_rate=0.001, model_dir=DIR_PATH+'model/', tensorboard=True, use_queue=False)
model.fit(train_dataset_balanced, nb_epoch=50)
model.get_checkpoints()

