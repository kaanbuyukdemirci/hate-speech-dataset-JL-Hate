from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import torch
from typing import Literal

try:
    from read_data import read_combined_data
    from commons import *
except:
    from source.read_data import read_combined_data
    from source.commons import *

def writer(path, combined_data, tokenizer, config, language:Literal['turkish', 'english']):
    # write dataframe
    combined_data.to_csv(f'{path}{language}_combined_data.tsv', sep='\t', index=True, header=True)

    # write text
    text = combined_data['text'].tolist()

    text = np.array(text)

    np.save(f'{path}{language}_text.npy', text)

    # tokenize
    tokenized_data = tokenizer.batch_encode_plus(combined_data['text'].tolist(), 
                                                 max_length=config['MAX_SEQUENCE_LENGTH'], 
                                                 padding=config['PADDING'], 
                                                 truncation=config['TRUNCATION'],
                                                 return_offsets_mapping=True,
                                                 add_special_tokens=config['ADD_SPECIAL_TOKENS'],
                                                 return_tensors='pt')

    # write sequence labels
    sequence_labels = combined_data['label'].tolist()

    sequence_labels = torch.tensor(sequence_labels).to(dtype=config['DTYPE'])

    torch.save(sequence_labels, f'{path}{language}_sequence_labels.pt')

    # write tokenized text
    tokenized_text = [tokenizer.convert_ids_to_tokens(tokenized_data['input_ids'].tolist()[i])
                      for i in range(len(combined_data))]

    tokenized_text = np.array(tokenized_text)

    np.save(f'{path}{language}_tokenized_text.npy', tokenized_text)

    # read spans and write token labels
    combined_data_with_no_text = combined_data.copy()
    #combined_data_with_no_text['text'] = ""
    combined_data_with_no_text['entities'].fillna("", inplace=True)
    spans = MajorityData.encoded_span_string_dataframe2span_list(combined_data_with_no_text)

    token_labels = [spans[i].calculate_token_labels(tokenized_data['offset_mapping'].tolist()[i])
                            for i in range(len(combined_data))]

    token_labels = torch.tensor(token_labels).to(dtype=config['DTYPE'])

    torch.save(token_labels, f'{path}{language}_token_labels.pt')
    
    # write input ids and attention masks
    torch.save(tokenized_data['attention_mask'], f'{path}{language}_attention_mask.pt')
    
    torch.save(tokenized_data['input_ids'], f'{path}{language}_input_ids.pt')

def write_tokenized_inputs(path):
    # write config
    config = {'TURKISH_MODEL': TURKISH_MODEL,
            'ENGLISH_MODEL': ENGLISH_MODEL,
            'MAX_SEQUENCE_LENGTH': MAX_SEQUENCE_LENGTH,
            'TRUNCATION': TRUNCATION,
            'PADDING': PADDING,
            'ADD_SPECIAL_TOKENS': ADD_SPECIAL_TOKENS,
            'DTYPE': DTYPE,
            'TIME': pd.Timestamp.now()}
    config = torch.nn.ParameterDict(config)
    torch.save(config, f'{path}config.pt')

    turkish_tokenizer = AutoTokenizer.from_pretrained(config['TURKISH_MODEL'])
    english_tokenizer = AutoTokenizer.from_pretrained(config['ENGLISH_MODEL'])

    # read
    combined_data = read_combined_data()
    combined_data = combined_data[(combined_data['label'] != ANNOTATION_SKIPPED) 
                                & (combined_data['label'] != ANNOTATION_NON_MAJOR)]
    turkish_combined_data = combined_data[combined_data['language'] == 0]
    english_combined_data = combined_data[combined_data['language'] == 1]

    writer(path, turkish_combined_data, turkish_tokenizer, config, 'turkish')
    writer(path, english_combined_data, english_tokenizer, config, 'english') 

if __name__ == '__main__':
    path = 'data/tokenized_input/'
    write_tokenized_inputs(path)