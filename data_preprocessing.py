'''
Vidya Sagar Kalvakunta (vkalvaku)
Subhash Bylaiah (sybylaiah)
'''

import os
import pandas as pd
import numpy as np
from scipy.stats.mstats import mode
from sklearn.preprocessing import LabelEncoder
from matplotlib.pyplot import rcParams

datadir = 'data'
result_datadir = 'data'


# Read file from the given path
def readFile(file_name):
    return pd.read_csv(os.path.join(datadir, file_name))


def process_merge_event_type(df, event_type):
    event_type = event_type.merge(df, on='id')
    unique_event_type = pd.DataFrame(event_type['event_type'].value_counts())
    # initialize:
    unique_event_type['preprocess'] = unique_event_type.index.values
    unique_event_type['TrainPerc'] = event_type.pivot_table(values='source', index='event_type',
                                                            aggfunc=lambda x: sum(x == 'train') / float(len(x)))

    # remove the ones not present in train:
    unique_event_type['preprocess'].loc[unique_event_type['TrainPerc'] == 0] = 'Delete'
    unique_event_type['severity_mode'] = event_type.loc[event_type['source'] == 'train'].pivot_table(
        values='fault_severity', index='event_type', aggfunc=lambda x: mode(x)[0])

    # replace the lower ones with mode:
    top_unchange = 33
    unique_event_type['preprocess'].iloc[top_unchange:] = unique_event_type['severity_mode'].iloc[top_unchange:].apply(
        lambda x: 'Delete' if pd.isnull(x) else 'event_type others_%d' % int(x))

    # Merge preprocess into original and then into train:
    event_type = event_type.merge(unique_event_type[['preprocess']], left_on='event_type', right_index=True)
    event_type_merge = event_type.pivot_table(values='event_type', index='id', columns='preprocess',
                                              aggfunc=lambda x: len(x), fill_value=0)
    return (df.merge(event_type_merge, left_on='id', right_index=True), event_type)


def process_merge_log_feature(df, log_feature):
    # log feature
    log_feature = log_feature.merge(df[['id', 'fault_severity', 'source']], on='id')
    uniuqe_log_feature = pd.DataFrame(log_feature['log_feature'].value_counts())
    uniuqe_log_feature['TrainPerc'] = log_feature.pivot_table(values='source', index='log_feature',
                                                              aggfunc=lambda x: sum(x == 'train') / float(len(x)))
    uniuqe_log_feature = pd.DataFrame(log_feature['log_feature'].value_counts())
    uniuqe_log_feature['TrainPerc'] = log_feature.pivot_table(values='source', index='log_feature',
                                                              aggfunc=lambda x: sum(x == 'train') / float(len(x)))
    # Determine the mode of each:
    uniuqe_log_feature['severity_mode'] = log_feature.loc[log_feature['source'] == 'train'].pivot_table(
        values='fault_severity', index='log_feature',
        aggfunc=lambda x: mode(x)[0])
    uniuqe_log_feature['preprocess'] = uniuqe_log_feature.index.values
    # remove the ones all in train
    uniuqe_log_feature['preprocess'].loc[uniuqe_log_feature['TrainPerc'] == 1] = np.nan
    top_unchange = 128
    uniuqe_log_feature['preprocess'].iloc[top_unchange:] = uniuqe_log_feature['severity_mode'].iloc[
                                                           top_unchange:].apply(
        lambda x: 'Delete' if pd.isnull(x) else 'feature others_%d' % int(x))
    log_feature = log_feature.merge(uniuqe_log_feature[['preprocess']], left_on='log_feature', right_index=True)
    log_feature_merge = log_feature.pivot_table(values='volume', index='id', columns='preprocess',
                                                aggfunc=np.sum, fill_value=0)
    return df.merge(log_feature_merge, left_on='id', right_index=True)


def process_merge_resource_type(df, resource_type):
    # resource type
    resource_type = resource_type.merge(df[['id', 'fault_severity', 'source']], on='id')
    unique_resource_type = pd.DataFrame(resource_type['resource_type'].value_counts())
    unique_resource_type['PercTrain'] = resource_type.pivot_table(values='source', index='resource_type',
                                                                  aggfunc=lambda x: sum(x == 'train') / float(len(x)))
    unique_resource_type.head()
    unique_resource_type['severity_mode'] = resource_type.loc[resource_type['source'] == 'train'].pivot_table(
        values='fault_severity', index='resource_type', aggfunc=lambda x: mode(x)[0])
    resource_type_merge = resource_type.pivot_table(values='source', index='id', columns='resource_type',
                                                    aggfunc=lambda x: len(x), fill_value=0)
    return df.merge(resource_type_merge, left_on='id', right_index=True)


def process_merge_severity_type(df, severity_type, event_type):
    # severity type
    severity_type = severity_type.merge(df[['id', 'fault_severity', 'source']], on='id')
    unique_severity_type = pd.DataFrame(severity_type['severity_type'].value_counts())
    unique_severity_type['PercTrain'] = severity_type.pivot_table(values='source', index='severity_type',
                                                                  aggfunc=lambda x: sum(x == 'train') / float(len(x)))
    unique_severity_type['severity_mode'] = severity_type.loc[severity_type['source'] == 'train'].pivot_table(
        values='fault_severity', index='severity_type', aggfunc=lambda x: mode(x)[0])
    severity_type.loc[event_type['source'] == 'train'].pivot_table(values='fault_severity', index='severity_type',
                                                                   aggfunc=lambda x: mode(x))
    severity_type_merge = severity_type.pivot_table(values='source', index='id', columns='severity_type',
                                                    aggfunc=lambda x: len(x), fill_value=0)
    return df.merge(severity_type_merge, left_on='id', right_index=True)


def main():
    train = readFile('train.csv')
    test = readFile('test.csv')
    # print "Test Data: ",test.head(10)
    event_type = readFile('event_type.csv')
    # print "event_type:", event_type.head(10)
    log_feature = readFile('log_feature.csv')
    # print "log_feature:", log_feature.head(10)
    resource_type = readFile('resource_type.csv')
    # print "resource_type:", resource_type.head(10)
    severity_type = readFile('severity_type.csv')
    # print "severity_type:", severity_type.head(10)
    train['source'] = 'train'
    test['source'] = 'test'
    df = pd.concat([train, test], ignore_index=True)
    # print "Imbalanced Class: ",df['fault_severity'].value_counts()
    # Process each file and merge df
    df, event_type = process_merge_event_type(df, event_type)
    df = process_merge_log_feature(df, log_feature)
    df = process_merge_resource_type(df, resource_type)
    df = process_merge_severity_type(df, severity_type, event_type)
    # print df.head(10)

    # Location Count:
    location_count = df['location'].value_counts()
    df['fualt_on_loc_count'] = df['location'].apply(lambda x: location_count[x])

    # Feature Count:
    features = [x for x in df.columns if 'feature ' in x]
    df['feature_count'] = df[features].apply(np.sum, axis=1)
    df['feature_count'].sum()

    # Convert location to numeric:
    encoder = LabelEncoder()
    df['location'] = encoder.fit_transform(df['location'])
    # Delete extra columns
    df.drop(['Delete_x', 'Delete_y'], axis=1, inplace=True)
    processed_train = df.loc[df['source'] == 'train']
    processed_test = df.loc[df['source'] == 'test']

    processed_train.drop('source', axis=1, inplace=True)
    processed_test.drop('source', axis=1, inplace=True)
    processed_train['num'] = processed_train.groupby('location')['fault_severity'].transform(
        lambda x: np.arange(x.shape[0]) + 1)
    processed_train['num'] = processed_train.groupby('location').cumcount() + 1
    # print processed_train.head(10)
    processed_test['num'] = processed_test.groupby('location')['fault_severity'].transform(
        lambda x: np.arange(x.shape[0]) + 1)
    processed_test['num'] = processed_test.groupby('location').cumcount() + 1

    processed_train.to_csv(os.path.join(result_datadir, 'processed_train1.csv'), index=False)
    processed_test.to_csv(os.path.join(result_datadir, 'processed_test1.csv'), index=False)
    print "Done!"

