import time
import uproot
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings(
    'ignore', category=pd.io.pytables.PerformanceWarning)


scaled_vars = [
        'Lepton_pt','Lepton_eta', 'LeadJet_pt', 'MET','ST','HT','nBTagMed_DeepFLV',
        'nCentralJets','DPHI_LepMet','DPHI_LepleadJet','DR_LepClosestJet',
        'bVsW_ratio','Angle_MuJet_Met'
]

selection_vars = []


sf = [0.00156816, # scaling factors
		0.000697108,
		0.000257874,
		2.77696e-05,
		1.91251e-05,
		4.33703e-06,
		9.15153e-08,
		0.00148474,
		6.8372e-07,
		2.29485e-06,
		5.0499e-07,
		4.12872e-05
]


def get_columns(fname):
    """
    Return variables to be kept and dropped. File specific reading
    can be done using the fname parameter.

    Parameters:
        fname (string): name of input file
    
    Returns:
        columns (list): columns to be read from the root file
        todrop (list): columns to be dropped after some processing
    """
    columns = scaled_vars + selection_vars + ['EvtWt']
    todrop = ['EvtWt', 'index']
    return columns, todrop


def build_filelist(input_dir):
    """
    Build a list of input files to be processed. Files are split into
    a few key groups. The systematics return value can be implemented
    later to process systematics uncertainty files separately.

    Parameters:
        input_dir (string): path to input files

    Returns:
        nominal (map[string]list): map file group to list of files
        systematics (map[string]list): empty map for now
    """
    files = [ifile for ifile in glob('{}/*.root'.format(input_dir))]

    nominal = {
        'tprime': [],  # in case you need to do anything special with these
        'wjets': [],
        'others': []
    }
    systematics = {}  # not used for now
    for fname in files:
        if 'WJets' in fname:
            nominal['wjets'].append(fname)
        elif 'Tprime' in fname:
            nominal['tprime'].append(fname)
        else:
            nominal['others'].append(fname)

    return nominal, systematics


def scaleing(weights,fname):
    if 'WJetsToLNu_HT-100To200' in fname:
        weights *= sf[0]
    elif 'WJetsToLNu_HT-200to400' in fname:
        weights *= sf[1]
    elif 'WJetsToLNu_HT-400To600' in fname:
        weights *= sf[2]
    elif 'WJetsToLNu_HT-600To800' in fname:
        weights *= sf[3]
    elif 'WJetsToLNu_HT-800To1200' in fname:
        weights *= sf[4]
    elif 'WJetsToLNu_HT-1200To2500' in fname:
        weights *= sf[5]
    elif 'WJetsToLNu_HT-2500ToInf' in fname:
        weights *= sf[6]
    elif 'TT_TuneCUETP8M2T4' in fname:
        weights *= sf[7]
    elif 'TTToSemiLeptonic' in fname:
        weights *= sf[8]
    elif 'TTTo2L2Nu' in fname:
        weights *= sf[9]
    elif 'TTToHadronic' in fname:
        weights *= sf[10]
    elif 'TprimeBToBW' in fname:
        weights *= sf[11] 
    return weights 


def process_files(all_data, files, is_signal):
    """
    Process input files, split into appropriate columns, do scaling, etc...

    Parameters:
        all_data (map[string]pandas.DataFrame): map from string to dataframes that are
        to be filled. The "meta" key is for meta-information (non-training variables) like
        the event weights, indexes, and unscaled selection variables. The "training" key
        is for variables to be scaled and used in NN training.
        files (list): list of input files
        is_signal (bool): label for if file is signal or not
    
    Returns:
        all_data (map[string]pandas.DataFrame): returns the provided map after filling
        the DataFrames with rows from the processed files
    """
    for ifile in files:
        print (ifile)
        columns, todrop = get_columns(ifile)  # if you need special branches from some files
        # open the TTree (named Skim) inside the file and read the specified columns into
        # a pandas dataframe. Also, get integral of hEventCount_wt for number of events.
        input_file = uproot.open(ifile)
        input_df = input_file['Skim'].pandas.df(columns)
        nevents = sum(input_file['Entry'].values)

        # set some file info
        filename = ifile.split('/')[-1].replace('.root', '')
        input_df['index'] = np.array([i for i in xrange(0, len(input_df))])  # event index before selection

        # do selection
        slim_df = input_df[(input_df['ST'] > 50)]  # keep only events: ST > 50
        # ...

        # clean our input data in case of mistakes in writing the tree
        slim_df = slim_df.dropna(axis=0, how='any')  # drop events with a NaN
        slim_df = slim_df.drop_duplicates()

        # Dataframe of variables used for selection, but not for training the
        # network. These variables are NOT rescaled.
        single_meta_df = pd.DataFrame(
            slim_df[selection_vars + ['index']].values,
            columns=slim_df[selection_vars + ['index']].columns.values
        )

        # add filenames to distinguish in training
        single_meta_df['names'] = np.full(len(slim_df), filename)

        # add signal vs. background labels
        isSignal = np.ones(len(slim_df)) if is_signal == 1 else np.zeros(len(slim_df))
        single_meta_df['isSignal'] = isSignal

        # Get the event weight and scale by xs if needed
        weight_df = scaleing(slim_df['EvtWt'], filename)

        # scale event weights between 1 - 2
        weight_df = MinMaxScaler(feature_range=(1., 2.)).fit_transform(
            weight_df.values.reshape(-1, 1))

        # save event weights
        single_meta_df['weights'] = weight_df

        # cleanup training variable dataframe
        single_training_df = slim_df.drop(selection_vars+todrop, axis=1)
        single_training_df = single_training_df.astype('float64')
        single_training_df['isSM'] = np.zeros(len(slim_df)) if is_signal == 1 else np.ones(len(slim_df))

        # combine with other files
        all_data['meta'] = pd.concat([all_data['meta'], single_meta_df])
        all_data['train'] = pd.concat([all_data['train'], single_training_df])
    return all_data


def build_scaler(sm_only):
    """
    Build StandardScaler for training variables, do the fit, and save the
    results to be used later.

    Parameters:
        sm_only (pandas.DataFrame): DataFrame of processed Standard Model files.
        Non-Standard Model events should not be included.

    Returns:
        scaler (StandardScaler): mean-0, variance-1 scaler already fit to SM
        scaler_info (pandas.DataFrame): information used to scale variables
    """
    scaler = StandardScaler()
    # only fit the nominal backgrounds
    scaler.fit(sm_only.values)
    scaler_info = pd.DataFrame.from_dict({
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'variance': scaler.var_,
        'nsamples': scaler.n_samples_seen_
    })
    scaler_info.set_index(sm_only.columns.values, inplace=True)
    return scaler, scaler_info


def format_for_store(all_data, scaler):
    """
    Get the processed DataFrames ready for storage in our dataset file. "meta"
    and "train" DataFrames are merged and the "train" variables are scaled.

    Parameters:
        all_data (map[string]pandas.DataFrame): all processed data
        scaler (StandardScaler): mean-0, variance-1 scaler to be applied to
        all_data['train]
    
    Returns:
        formatted_data (pandas.DataFrame): merged DataFrame with all events/variables
        included and scaled appropriately.
    """
    # apply scaling to all samples
    formatted_data = pd.DataFrame(
        scaler.transform(all_data['train'].values),
        columns=all_data['train'].columns.values, dtype='float64'
    )

    formatted_data['idx'] = all_data['meta']['index'].values
    formatted_data['sample_names'] = all_data['meta']['names'].values
    formatted_data['signal'] = all_data['meta']['isSignal'].values
    formatted_data['EvtWt'] = all_data['meta']['weights'].values
    return formatted_data


def main(args):
    start = time.time()

    # create the store
    store = pd.HDFStore('datasets/{}.h5'.format(args.output),
                        complevel=9, complib='bzip2')

    all_data = {
        'meta': pd.DataFrame(),
        'train': pd.DataFrame()
    }
    filelist, _ = build_filelist(args.input)  # list of files to process

    # process all files
    all_data = process_files(all_data, filelist['tprime'], is_signal=1)
    all_data = process_files(all_data, filelist['wjets'], is_signal=0)
    all_data = process_files(all_data, filelist['others'], is_signal=0)

    # build scaler using SM only
    sm_only = all_data['train'][(all_data['train']['isSM'] == 1)]
    scaler, store['scaler'] = build_scaler(sm_only)

    # format data and store
    store['nominal'] = format_for_store(all_data, scaler)

    print ('Complete! Preprocessing completed in {} seconds'.format(time.time() - start))


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='path to input files')
    parser.add_argument('--output', '-o', required=True, help='name of output file')
    main(parser.parse_args())
