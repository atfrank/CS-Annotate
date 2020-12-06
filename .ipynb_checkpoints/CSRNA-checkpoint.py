import rdkit as rd
import deepchem as dc
import numpy as np
import tensorflow as tf
import pandas as pd

NUMBER_CHEMICAL_SHIFT_TYPE = 19
RETAIN = ['id', 'resid', 'resname', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']
RETAIN_NONAME = ['id', 'resid', 'sasa-All-atoms', 'sasa-Total-Side', 'sasa-Main-Chain', 'sasa-Non-polar', 'sasa-All', 'sasa', 'syn_anti', 'astack', 'nastack', 'pair', 'pucker', 'class']

def get_cs_all(cs_all, id):
        '''
        This function gets chemical shifts for a particular RNA.
        '''
        return(cs_all[cs_all.id == id])

def get_cs_residues(cs_i, resid, dummy=0):
        '''
        This function return an array contining the chemical shifts 
        for a particular residues in an RNA.
        '''
        cs_tmp=cs_i[(cs_i.resid == resid)].drop(RETAIN, axis=1)
        info_tmp=cs_i[(cs_i.resid == resid)]
        if (cs_tmp.shape[0] != 1):
                return(dummy*np.ones(shape=(1, NUMBER_CHEMICAL_SHIFT_TYPE)))
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

def get_cs_features(cs_i, resid, neighbors):
        '''
        This function return chemical shifts and resnames for 
        residues (resid) and its neighbors
        '''
        cs=[]
        resnames=[]
        for i in range(resid-neighbors, resid+neighbors+1):
                cs.append(get_cs_residues(cs_i, i))
                resnames.append(get_resnames(cs_i, i))
        return(resnames, np.array(cs))

def get_columns_name(neighbors=3, chemical_shift_types = NUMBER_CHEMICAL_SHIFT_TYPE):
        '''
        Helper function that writes out the required column names
        '''
        #tmp=2*neighbors+1
        #neighbors=1
        columns=RETAIN
        for i in range(0, neighbors*NUMBER_CHEMICAL_SHIFT_TYPE):
                columns.append(i)
        return(columns)

def write_out_resname(neighbors=1):
        ''' 
        Helper function that writes out the column names associated 
        resnames for a given residue and its neighbors
        '''  
        colnames = []
        for i in range(1-neighbors-1, neighbors+1):
                if i < 0: 
                        colnames.append('R%s'%i)
                elif i > 0: 
                        colnames.append('R+%s'%i)
                else: 
                        colnames.append('R')
        return(colnames)        

def get_cs_features_rna(cs, neighbors=1):
        '''      
        This function generates the complete required data frame an RNA      
        '''
        all_features = []
        all_resnames = []
        for resid in cs['resid'].unique():
                resnames, features = get_cs_features(cs, resid, neighbors)
                all_features.append(features.flatten())
                all_resnames.append(resnames)

        all_resnames = pd.DataFrame(all_resnames, dtype='object', columns = write_out_resname(neighbors))
        all_features = pd.DataFrame(all_features, dtype='object')
        info = pd.DataFrame(cs[RETAIN_NONAME].values, dtype='object', columns = RETAIN_NONAME)
        return(pd.concat([info, all_resnames, all_features], axis=1))

def get_cs_features_rna_all(cs, neighbors):  
        '''      
        This function generate a pandas dataframe containing training data for all RNAs
        Each row in the data frame should contain the class and chemical shifts for given residue and neighbors in a given RNA.
        '''  
        cs_new=pd.DataFrame()
        for pdbid in cs['id'].unique()[0 :]:
                tmp=get_cs_features_rna(get_cs_all(cs, id=pdbid), neighbors)
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

def get_data(neighbors, training = True):
    ''' 
    Collects and combines data for training and testing the model
    '''
    # load sasa data
    if training:        
        fname="data/final_training_sasa.csv"
    else:       
        fname="data/final_testing_sasa.csv"
    sasa=pd.read_csv(fname, sep = "\s+")
    print("[INFO]: SASA loaded data")

    # load cs data
    fname="data/chemical_shifts.csv"
    cs=pd.read_csv(fname, sep = " ")
    print("[INFO]: CS loaded data")

    # load MC-Annotate
    fname="data/annotations.csv"
    mc_anontate=pd.read_csv(fname, sep = " ")
    print("[INFO]: MC-Annotate loaded data")

    # merge sasa and cs
    data = pd.merge(cs, sasa, on=['id', 'resname', 'resid'])

    # merge with mc-annotate structure
    data = pd.merge(data, mc_anontate, on=['id', 'resname', 'resid'])    
    drop_names = ['sugar_puckering', 'pseudoknot', 'junk']      
    print("[INFO]: merged SASA, CS, and MC-Annotate")

    # prepare for testing
    data_all = get_cs_features_rna_all(data, neighbors = neighbors)
    print("[INFO]: Prepared final data set")

    # one-hot-encode data
    data_all = one_hot_encode(data_all, write_out_resname(neighbors))
    print("[INFO]: One-hot encoded data")
    return(data_all)

def process_data(train, test):
    '''
    Readies data for Deepchem
    '''
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
    
    # testing
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
    actuals = pd.DataFrame(testy_mix, columns=targets)

    targets = ["sasa","astack","nastack","pair","syn_anti","pucker_C2p_endo","pucker_C3p_endo","pucker_C2p_exo","pucker_C3p_exo","pucker_C1p_exo","pucker_C4p_exo"]
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
                 "pucker_C4p_exo":trainy_mix[:, 10]}, columns = targets).to_csv("data/train_target.csv", sep = " ", index = False)

    trainX.to_csv("data/train_features.csv", sep = " ", index = False)

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
                 "pucker_C4p_exo":testy_mix[:, 10]}, columns = targets).to_csv("data/test_target.csv", sep = " ", index = False)

    testX.to_csv("data/test_features.csv", sep = " ", index = False)

    train_info = pd.DataFrame(train[retain].values, dtype='object', columns = retain)
    test_info = pd.DataFrame(test[retain].values, dtype='object', columns = retain)
    
    
    return(train_dataset, test_dataset, train_info, test_info, targets)
