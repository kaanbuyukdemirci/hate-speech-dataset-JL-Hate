import pandas as pd
import os

try:
    from commons import *
except ModuleNotFoundError:
    from source.commons import *


def read_initial_data(path='data/initial_data/'):
    dataframes = os.listdir(path)
    dataframes = [path + x for x in dataframes]
    dataframes = pd.concat([pd.read_csv(x, sep='\t', index_col='tweet_id', header=0) for x in dataframes])
    return dataframes

def read_annotated_data(path='data/annotated_data/'):
    annotator_label_names = [x + '_labels' for x in ANNOTATOR_LABEL_NAMES]
    dataframes = os.listdir(path)
    dataframes = [path + x for x in dataframes]
    df = []
    for x in dataframes:
        x = pd.read_csv(x, sep='\t', index_col='tweet_id', header=0)
        included_annotators = [y for y in ANNOTATOR_LABEL_NAMES if y in x.columns]
        x[included_annotators] = x[included_annotators].fillna(-2)
        df.append(x)
    df = pd.concat(df)
    included_annotators = [x for x in ANNOTATOR_LABEL_NAMES if x in df.columns]
    df[ANNOTATOR_LABEL_NAMES] = df[ANNOTATOR_LABEL_NAMES].fillna(-1)
    #TODO: there is an error in emre and oguzhan's annotations. Make them empty
    error_columns = ['emre_labels', 'oguzhan_labels']
    df[error_columns] = ""
    
    #TODO: there is an error in one of cagri's annotation. Make them empty
    df.loc[1278062838184017922, 'cagri_labels'] = ''
    return df

def read_majority_annotated_data(path='data/majority_annotated_data/'): #TODO: unreliable - create this again by using majority algorithm
    dataframes = os.listdir(path)
    dataframes = [path + x for x in dataframes]
    dataframes = pd.concat([pd.read_csv(x, sep='\t', index_col='tweet_id', header=0) for x in dataframes])
    return dataframes

def read_unknown_data(path='data/unknown_data/'):
    dataframes = os.listdir(path)
    dataframes = [path + x for x in dataframes]
    dataframes = [[x, pd.read_csv(x, sep='\t', index_col='tweet_id', header=0)] for x in dataframes]
    return dataframes

def read_combined_data(path='data/combined_data/'):
    dataframes = os.listdir(path)
    dataframes = [path + x for x in dataframes]
    dataframes = pd.concat([pd.read_csv(x, sep='\t', index_col='tweet_id', header=0) for x in dataframes])
    dataframes['text'] = dataframes['text'].str.replace("\r", "")
    return dataframes

def read_dataset(path='data/tokenized_input/'):
    config = torch.load(f'{path}config.pt')
    
    turkish_input_ids = torch.load(f'{path}turkish_input_ids.pt').to(device=DEVICE, dtype=config['DTYPE'])
    turkish_attention_mask = torch.load(f'{path}turkish_attention_mask.pt').to(device=DEVICE, dtype=config['DTYPE'])
    turkish_sequence_labels = torch.load(f'{path}turkish_sequence_labels.pt').to(device=DEVICE, dtype=config['DTYPE'])
    turkish_token_labels = torch.load(f'{path}turkish_token_labels.pt').to(device=DEVICE, dtype=config['DTYPE'])
    turkish_tokenized_text = np.load(f'{path}turkish_tokenized_text.npy')
    turkish_text = np.load(f'{path}turkish_text.npy')
    turkish_text = np.array([x.replace("\r", "") for x in turkish_text])
    turkish_ignore_indexes = (turkish_sequence_labels == IGNORE_INDEX) | (turkish_sequence_labels == ANNOTATION_NON_MAJOR)
    
    english_input_ids = torch.load(f'{path}english_input_ids.pt').to(device=DEVICE, dtype=config['DTYPE'])
    english_attention_mask = torch.load(f'{path}english_attention_mask.pt').to(device=DEVICE, dtype=config['DTYPE'])
    english_sequence_labels = torch.load(f'{path}english_sequence_labels.pt').to(device=DEVICE, dtype=config['DTYPE'])
    english_token_labels = torch.load(f'{path}english_token_labels.pt').to(device=DEVICE, dtype=config['DTYPE'])
    english_tokenized_text = np.load(f'{path}english_tokenized_text.npy')
    english_text = np.load(f'{path}english_text.npy')
    english_text = np.array([x.replace("\r", "") for x in english_text])
    english_ignore_indexes = (english_sequence_labels == IGNORE_INDEX) | (english_sequence_labels == ANNOTATION_NON_MAJOR)
    
    turkish_dataset = HateDataset(turkish_input_ids[~turkish_ignore_indexes], 
                                  turkish_attention_mask[~turkish_ignore_indexes], 
                                  turkish_sequence_labels[~turkish_ignore_indexes], 
                                  turkish_token_labels[~turkish_ignore_indexes], 
                                  turkish_tokenized_text[~turkish_ignore_indexes.numpy()], 
                                  turkish_text[~turkish_ignore_indexes.numpy()], 
                                  config)
    english_dataset = HateDataset(english_input_ids[~english_ignore_indexes], 
                                  english_attention_mask[~english_ignore_indexes], 
                                  english_sequence_labels[~english_ignore_indexes], 
                                  english_token_labels[~english_ignore_indexes], 
                                  english_tokenized_text[~english_ignore_indexes.numpy()], 
                                  english_text[~english_ignore_indexes.numpy()], 
                                  config)
    return turkish_dataset, english_dataset

if __name__ == '__main__':
    # print neutral tweet in english and turkish
    df = read_majority_annotated_data()
    lng_filt = df['language']==0
    lbl_filt = df['label']==0
    df_tr = df[lng_filt & lbl_filt]
    df_en = df[~lng_filt & lbl_filt]
    i = 0
    while True:
        print("-------------------------------------------------------")
        #print(df_tr.iloc[i]['text'])
        print(df_en.iloc[i]['text'])
        i += 1
        input()
    