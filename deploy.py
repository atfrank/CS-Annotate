# other
import numpy as np
import pandas as pd 
import requests
import io
import argparse

RETAIN = ['id', 'resid', 'resname']
RETAIN_NONAME = []

def format_cs(input_file, rna = "none"):
  import pandas as pd
  cs = pd.read_csv(input_file, delim_whitespace=True, header=None, names = ['resname', 'resid', 'nucleus', 'cs', 'error'])
  df = []
  resid = cs['resid'].unique()
  nuclei = ["C1'","C2'","C3'","C4'","C5'","C2","C5","C6","C8","H1'","H2'","H3'","H4'","H2","H5","H5'","H5''","H6","H8"]
  for i in resid:
    resname = cs.loc[cs['resid']==i,"resname"].values[0]
    cs_residue = []
    for j in nuclei:
      try:
        cs_residue.append(cs.loc[(cs['nucleus']==j) & (cs['resid']==i),"cs"].values[0])
      except:
        cs_residue.append(None)
    features = [rna, i, resname]+cs_residue
    df.append(features)

  df = pd.DataFrame(df)
  df.columns = ["id","resid","resname"]+list(map(lambda x: x.replace("'","p"), nuclei))
  return(df)
  
def get_cs_all(cs_all, id):
    '''
    This function gets chemical shifts for a particular RNA.
    '''
    return(cs_all[cs_all.id == id])

def get_cs_residues(cs_i, resid, dummy=0, number_of_cs_types=19):
    '''
    This function return an array contining the chemical shifts 
    for a particular residues in an RNA.
    '''
    cs_tmp=cs_i[(cs_i.resid == resid)].drop(RETAIN, axis=1)
    info_tmp=cs_i[(cs_i.resid == resid)]
    if (cs_tmp.shape[0] != 1):
        return(dummy*np.ones(shape=(1, number_of_cs_types)))
    else:
        return(cs_tmp.values)

def get_resnames(cs_i, resid, dummy="UNK"):
    '''
    This function returns the residue name for specified residue (resid)
    '''
    cs_tmp=cs_i[(cs_i.resid == resid)]
    if (cs_tmp.shape[0] != 1):
        return(dummy)
    else:
        return(cs_tmp['resname'].values[0])

def get_cs_features(cs_i, resid, neighbors, number_of_cs_types):
    '''
    This function return chemical shifts and resnames for 
    residues (resid) and its neighbors
    '''
    cs=[]
    resnames=[]
    for i in range(resid-neighbors, resid+neighbors+1):
        cs.append(get_cs_residues(cs_i, resid=i, number_of_cs_types=number_of_cs_types))
        resnames.append(get_resnames(cs_i, i))
    return(resnames, np.array(cs))

def write_out_resname(neighbors=1):
    ''' 
    Helper function that writes out the column names associated 
    resnames for a given residue and its neighbors
    '''  
    colnames = []
    for i in range(1-neighbors-1, neighbors+1):  # ['R-2', 'R-1', 'R', 'R+1', 'R+2'] when neighbors = 2
        if i < 0: 
            colnames.append('R%s'%i)
        elif i > 0: 
            colnames.append('R+%s'%i)
        else: 
            colnames.append('R')
    return(colnames)    

def get_cs_features_rna(cs, neighbors, number_of_cs_types):
    '''    
    This function generates the complete required data frame an RNA    
    '''
    all_features = []
    all_resnames = []
    for resid in cs['resid'].unique():
        resnames, features = get_cs_features(cs, resid, neighbors, number_of_cs_types)
        all_features.append(features.flatten())
        all_resnames.append(resnames)

    all_resnames = pd.DataFrame(all_resnames, dtype='object', columns = write_out_resname(neighbors))
    all_features = pd.DataFrame(all_features, dtype='object')
    info = pd.DataFrame(cs[RETAIN_NONAME].values, dtype='object', columns = RETAIN_NONAME)
    return(pd.concat([info, all_resnames, all_features], axis=1))

def get_cs_features_rna_all(cs, neighbors, number_of_cs_types):  
    '''    
    This function generate a pandas dataframe containing training data for all RNAs
    Each row in the data frame should contain the class and chemical shifts for given residue and neighbors in a given RNA.
    '''  
    cs_new=pd.DataFrame()
    for pdbid in cs['id'].unique()[0 :]:
        tmp=get_cs_features_rna(get_cs_all(cs, id=pdbid), neighbors, number_of_cs_types)
        cs_new=pd.concat([cs_new, tmp], axis=0)
    return(cs_new)
 
def one_hot_encode(df, hot_columns):
  '''
    This function generate one hot encodes a dataFrame
    see: http://queirozf.com/entries/one-hot-encoding-a-feature-on-a-pandas-dataframe-an-example 
  '''
  for hot_column in hot_columns:
    # use pd.concat to join the new columns with your original dataframe
    df = pd.concat([df, pd.get_dummies(df[hot_column], prefix=hot_column)],axis=1)
    # now drop the original 'country' column (you don't need it anymore)
    df.drop([hot_column],axis=1, inplace=True)
  return(df)

def balance_transformer(dataset):
    '''
      Copy of deepchem function for reweighting samples. 
      Deepchem version does not work.
    '''
    # Compute weighting factors from dataset.
    y = dataset.y
    w = dataset.w
    # Ensure dataset is binary
    np.testing.assert_allclose(sorted(np.unique(y)), np.array([0., 1.]))
    weights = []
    for ind, task in enumerate(dataset.get_task_names()):
      task_w = w[:]
      task_y = y[:]
      # Remove labels with zero weights
      task_y = task_y[task_w != 0]
      num_positives = np.count_nonzero(task_y)
      num_negatives = len(task_y) - num_positives
      if num_positives > 0:
        pos_weight = float(num_negatives) / num_positives
      else:
        pos_weight = 1
      neg_weight = 1
      weights.append((neg_weight, pos_weight)) 
    return(dc.data.NumpyDataset(X=dataset.X, y=dataset.y, w=np.where(y==1, pos_weight, neg_weight)))

def get_data(neighbors, training = True, partial = None):
  # load sasa data
  if training:    
    url="https://drive.google.com/uc?id=1Y3Imx-lTjGKCQAFqEKTbaMSzFARtwEFN&authuser=afrankz@umich.edu&usp=drive_fs"
  else:   
    url="https://drive.google.com/uc?id=1jLcowU89y4o5Xmv_qBs3VgZre5ZFFYwG&authuser=afrankz@umich.edu&usp=drive_fs"

  s=requests.get(url).content
  sasa=pd.read_csv(io.StringIO(s.decode('utf-8')), sep = "\s+")
  print("[INFO]: SASA loaded data")
  print(sasa.head())

  # load cs data
  url="https://drive.google.com/uc?id=1ApGAKHnzKUjri-f_sSPZwqK5N3Cr7gPq&authuser=afrankz@umich.edu&usp=drive_fs" 
  s=requests.get(url).content
  cs=pd.read_csv(io.StringIO(s.decode('utf-8')), sep = " ")

  # if partial = None, use both carbon and proton 
  # if partial = "C", use carbon chemical shifts only
  # if partial = "H", use proton chemical shifts only
  if partial == "C":
    cs.drop(columns=['H1p','H2p','H3p','H4p','H2','H5','H5p','H5pp','H6','H8'],inplace=True)
    number_of_cs_types = 9

  if partial == "H":
    cs.drop(columns=['C1p','C2p','C3p','C4p','C5p','C2','C5','C6','C8'],inplace=True)
    number_of_cs_types = 10

  if partial == None:
    number_of_cs_types = 19
  print("[INFO]: CS loaded data")
  #print(cs.head())

  # load MC-Annotate
  url="https://drive.google.com/uc?id=15dL0aZJZpTRxPYQn-DQ_jKC7tlH7fejg"
  s=requests.get(url).content
  mc_annotate=pd.read_csv(io.StringIO(s.decode('utf-8')), sep = " ")
  #print("[INFO]: MC-Annotate loaded data")
  
  # merge sasa and cs
  data = pd.merge(cs, sasa, on=['id', 'resname', 'resid'])
  # merge with mc-annotate structure
  data = pd.merge(data, mc_annotate, on=['id', 'resname', 'resid'])  
  #drop_names = ['sugar_puckering', 'pseudoknot', 'junk']
  #data = data.drop(drop_names, axis=1)
  print("[INFO]: merged SASA, CS, and MC-Annotate")
  #print(data.head())

    # prepare for testing
  print(data.columns)
  data_all = get_cs_features_rna_all(data, neighbors = neighbors, number_of_cs_types = number_of_cs_types)
  print("[INFO]: Prepared final data set")
  #print(data_all.head())

  data_all = one_hot_encode(data_all, write_out_resname(neighbors)) # only encode resnames (including neighbors)
  print("[INFO]: One-hot encoded data")
  #print(data_all.head())
  return(data_all)

def get_test_from_file(input_file):    
    data = format_cs(input_file = input_file, rna = "none")
    info = data[['id','resid', 'resname']]
    data = get_cs_features_rna(data, neighbors=3, number_of_cs_types = 19)
    data = one_hot_encode(data, write_out_resname(neighbors=3)) # only encode resnames (including neighbors)
    return (data, info)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--infile", help="file chemical shift map", required = True)
    parser.add_argument("-o","--outfile", help="file chemical shift map", default = "data/cs-annotate-predictions.txt")
    parser.add_argument("-m","--model", help="path to saved model", default = "models/one-layer/")
    parser.add_argument("-X","--Xtrain", help="X data used to train model", default = "data/train_features.csv")
    parser.add_argument("-y","--ytrain", help="y data used to train model", default = "data/train_target.csv")
    
    a = parser.parse_args()
    
    import rdkit as rd
    import deepchem as dc
    import tensorflow as tf

    # set neighbors (hard coded because it gives the best results)
    neighbors = 3
    
    # load train and test data
    X_train = pd.read_csv(a.Xtrain,delim_whitespace=True,header=0)
    y_train = pd.read_csv(a.ytrain,delim_whitespace=True,header=0)
    targets = y_train.columns
    
    # read in test data
    test, info = get_test_from_file(input_file = a.infile)
    X_test = test.values
    y_test = np.ones(y_train.shape[0])    

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

    # load model
    model = dc.models.ProgressiveMultitaskClassifier(n_tasks=n_tasks,n_features=n_features,layer_sizes=[100],alpha_init_stddevs=0.04,learning_rate=0.001, model_dir=a.model, tensorboard=True, use_queue=False)
    model.restore()
    
    # predict
    testpred = model.predict(test_dataset_norm)
    testpred = pd.DataFrame(testpred[:, :, 1], columns=["p"+i for i in targets])
    testpred = pd.concat([info, testpred], axis=1)

    # output
    pd.options.display.float_format = '{:,.2f}'.format
    testpred.to_csv(a.outfile, index = False)
    print(testpred)
