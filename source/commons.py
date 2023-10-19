import re
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import transformers
from typing import Literal
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall

EMOJI_REGEX_PATTERN = re.compile(pattern = "["
                                 u"\U0001F600-\U0001F64F"  # emoticons
                                 u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                 u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 "]+", flags = re.UNICODE)
PRIORITY_DICTIONARY = {'HTAR': 4, 'HSIG': 3, 'OTAR': 2, 'OSIG': 1, 'O': 0}
TOKEN_DICTIONARY = PRIORITY_DICTIONARY
ANNOTATION_SKIPPED = -2
ANNOTATOR_SKIPPED = -2
ANNOTATION_NON_MAJOR = -1
ANNOTATOR_NOT_SEEN = -1

ANNOTATOR_LABEL_NAMES = ['cagri', 'emre', 'kaan', 'oguzhan', 'umitcan']
ANNOTATOR_ANNOTATION_NAMES = [x + '_labels' for x in ANNOTATOR_LABEL_NAMES]
NUMBER_OF_ANNOTATORS = len(ANNOTATOR_LABEL_NAMES)

LABEL_DICTIONARY = {'skipped':-2, 'tie':-1,'neutral':0, 'offensive':1, 'hateful':2}

TURKISH_MODEL = 'dbmdz/convbert-base-turkish-cased' #'dbmdz/distilbert-base-turkish-cased'
ENGLISH_MODEL = 'distilroberta-base'

CHECKPOINT = 'checkpoints'
TURKISH_CHECK_POINT = f'{CHECKPOINT}/turkish_checkpoint'
ENGLISH_CHECK_POINT = f'{CHECKPOINT}/english_checkpoint'

MAX_SEQUENCE_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
EPSILON = 1e-8
TRUNCATION = True
PADDING = 'max_length'
ADD_SPECIAL_TOKENS = True

DTYPE = torch.int64
DEVICE = torch.device('cpu') #torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

IGNORE_INDEX = -100

RESULT_TYPE:Literal['best', 'last'] = 'best'
TAG_TYPE:Literal['BIO', 'IO'] = 'IO'

def deep_list_len(lis):
        return sum(list(map(len, lis)))

def count_emojis(text):
    return deep_list_len(re.findall(EMOJI_REGEX_PATTERN, text))

def duplicate_emojis(text):
        return EMOJI_REGEX_PATTERN.sub(lambda match: match.group(0)*2, text)


class Span(object):
    """The Span class is used to represent spans of text. It is assumed that intervals do not overlap.
    """
    def __init__(self, intervals:list[tuple[int,int]], priorities:list[int], labels:list[str], text:str=None):
        """The constructor of the Span class.

        Parameters
        ----------
        intervals : list[tuple[int,int]]
            The intervals of the span. Each interval is a tuple of two integers, the first one being the start index 
            and the second one being the end index. It is assumed that intervals do not overlap.
        priority : list[int]
            The priorities of the intervals. Each interval has a priority. Each priority is an integer.
        label : list[str]
            The labels of the intervals. Each interval has a label. Each label is a string.
        text : str, optional
            The text of the span. The default is None.
        """
        self.intervals = intervals
        self.priorities = priorities
        self.labels = labels
        self.text = text
    
    def __add__(self, other):
        """The intersection operator of the Span class. It is assumed that the different Span objects with the same priority have the same label.
        It is also assumed that the self and other Span objects have the same text.
        The rule to follow is as follows:
            If there is an intersection between two intervals, and their priorities are not equal, choose the one with the higher priority. 
                The label of the chosen interval will be the label of the higher priority interval. 
                The priority of the chosen interval will be the higher priority.
            If there is an intersection between two intervals, and their priorities are equal, combine both of them.
                The label of the combined interval will be the common label. 
                The priority of the combined interval will be the common priority.
            If there is no intersection between two intervals, add both of them.
                The labels of the chosen intervals will be the same as before.
                The priorities of the chosen intervals will be the same as before.

        Parameters
        ----------
        other : Span
            The other span to intersect with.

        Returns
        -------
        Span
            The intersection of the two spans.
        """
        intervals = []
        priorities = []
        labels = []
        
        intervals1 = self.intervals
        priorities1 = self.priorities
        labels1 = self.labels
        len1 = len(intervals1)
        
        intervals2 = other.intervals
        priorities2 = other.priorities
        labels2 = other.labels
        len2 = len(intervals2)
        
        unique_priorities = list(set(priorities1 + priorities2))
        unique_priorities.sort()
        unique_priorities.reverse()
        unique_labels = {}
        for i in range(len(unique_priorities)):
            if unique_priorities[i] in priorities1:
                index = priorities1.index(unique_priorities[i])
                label = labels1[index]
            else:
                index = priorities2.index(unique_priorities[i])
                label = labels2[index]
            unique_labels[unique_priorities[i]] = label
        
        unique_priority_indexes1 = [[i for i in range(len1) if priorities1[i] == unique_priority] for unique_priority in unique_priorities]
        unique_priority_indexes2 = [[i for i in range(len2) if priorities2[i] == unique_priority] for unique_priority in unique_priorities]
        
        for i in range(len(unique_priorities)):
            priority_i_intervals1 = [intervals1[j] for j in unique_priority_indexes1[i]]
            priority_i_len1 = len(priority_i_intervals1)
            priority_i_intervals2 = [intervals2[j] for j in unique_priority_indexes2[i]]
            priority_i_len2 = len(priority_i_intervals2)
            priority_i_label = unique_labels[unique_priorities[i]]
            priority_i_priority = unique_priorities[i]
            
            # delete intervals from priority_i_intervals1 and priority_i_intervals2 that intersect with intervals from intervals
            for j in range(priority_i_len1-1, -1, -1): # iterate backwards to avoid index errors
                intersection = False
                for interval in intervals:
                    if (priority_i_intervals1[j][0] <= interval[0] <= priority_i_intervals1[j][1]) or\
                        (interval[0] <= priority_i_intervals1[j][0] <= interval[1]):
                            intersection = True
                if intersection:
                    priority_i_intervals1.pop(j)
                    priority_i_len1 -= 1
            for j in range(priority_i_len2-1, -1, -1): # iterate backwards to avoid index errors
                intersection = False
                for interval in intervals:
                    if (priority_i_intervals2[j][0] <= interval[0] <= priority_i_intervals2[j][1]) or\
                        (interval[0] <= priority_i_intervals2[j][0] <= interval[1]):
                            intersection = True
                if intersection:
                    priority_i_intervals2.pop(j)
                    priority_i_len2 -= 1
        
            # add intervals from priority_i_intervals1 and priority_i_intervals2 to intervals
            intersections = [False for _ in range(priority_i_len2)]
            for interval1 in priority_i_intervals1:
                intersection = False
                for j, interval2 in enumerate(priority_i_intervals2):
                    # If there is an intersection between two intervals, and their priorities are equal
                    if (interval1[0] <= interval2[0] <= interval1[1]) or (interval2[0] <= interval1[0] <= interval2[1]):
                        # combine both of them
                        intervals.append((min(interval1[0], interval2[0]), max(interval1[1], interval2[1])))
                        priorities.append(priority_i_priority)
                        labels.append(priority_i_label)
                        intersections[j] = True
                        intersection = True
                # If there is no intersection between two intervals
                if not intersection:
                    # add both of them (1 here)
                    intervals.append(interval1)
                    priorities.append(priority_i_priority)
                    labels.append(priority_i_label)
            for j, interval2 in enumerate(priority_i_intervals2):
                # If there is no intersection between two intervals
                if not intersections[j]:
                    # add both of them (1 here)
                    intervals.append(interval2)
                    priorities.append(priority_i_priority)
                    labels.append(priority_i_label)
    
        # return
        return Span(intervals, priorities, labels, self.text)
        
    def __repr__(self) -> str:
        """The representation of the Span class.

        Returns
        -------
        str
            The representation of the Span class.
        """
        if self.text is None:
            return_text = f"Span({self.intervals}, {self.priorities}, {self.labels})"
        else:
            return_text = f"Span({self.intervals}, {self.priorities}, {self.labels}, {self.text})"
            for interval, priority, label in zip(self.intervals, self.priorities, self.labels):
                return_text += '\n'
                return_text += f"interval:{interval}, priority:{priority}, label:{label}, text:{self.text[interval[0]:interval[1]]}"
        return return_text
        
    def calculate_statistics(self):
        # unique labels
        unique_labels = list(set(self.labels))
        
        # number of intervals per unique label
        number_of_intervals_per_unique_label = [self.labels.count(unique_label) for unique_label in unique_labels]
        
        # number of word counts per unique label
        number_of_word_counts_per_unique_label = [len(re.findall(r'\w+', self.text[interval[0]:interval[1]])) for interval in self.intervals]
        
        # return
        return unique_labels, number_of_intervals_per_unique_label, number_of_word_counts_per_unique_label

    def calculate_token_labels(self, offset_mapping:list[tuple[int,int]]):
        token_labels = []
        for offset_map in offset_mapping:
            if offset_map[0] == 0 and offset_map[1] == 0:
                token_labels.append(IGNORE_INDEX)
            else:
                
                for interval, label in zip(self.intervals, self.labels):
                    if  (interval[0] <= offset_map[0] < interval[1]) or (offset_map[0] <= interval[0] < offset_map[1]):
                        token_labels.append(TOKEN_DICTIONARY[label])
                        break
                else:
                    token_labels.append(TOKEN_DICTIONARY['O'])
        
        # add len(TOKEN_DICTIONARY.keys())-1 to differentiate between B and I labels
        if TAG_TYPE == 'BIO':
            for i in range(len(token_labels)-1, 0, -1):
                if (token_labels[i] != token_labels[i-1]) and (token_labels[i] != TOKEN_DICTIONARY['O']) and (token_labels[i] != IGNORE_INDEX):
                    token_labels[i] += len(TOKEN_DICTIONARY.keys())-1
            if (token_labels[0] != TOKEN_DICTIONARY['O']) and (token_labels[0] != IGNORE_INDEX):
                token_labels[0] += len(TOKEN_DICTIONARY.keys())-1
        
        return token_labels
        
class MajorityData(object):
    def __init__(self, dataframe:pd.DataFrame, duplicated_emojis:bool=True):
        """The constructor of the MajorityData class.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataframe to be used.
        duplicated_emojis : bool, optional
            Some annotated data is stored in a way that counted emojis as 2 characters, which causes indexes to shift. 
            This parameter is used to duplicate emojis in the text to match the annotated data, which negates the problem. 
            The default is True.
        """
        self.dataframe = dataframe.fillna('')
        self.duplicated_emojis = duplicated_emojis
        self.emoji_regex_pattern = EMOJI_REGEX_PATTERN
        self.__init_annotators()

    def __init_annotators(self):
        self.annotator_label_names = []
        self.annotator_annotation_names = []
        for annotator_label_name in ANNOTATOR_LABEL_NAMES:
            if annotator_label_name in self.dataframe.columns:
                self.annotator_label_names.append(annotator_label_name)
        for annotator_annotation_name in ANNOTATOR_ANNOTATION_NAMES:
            if annotator_annotation_name in self.dataframe.columns:
                self.annotator_annotation_names.append(annotator_annotation_name)

    def calculate_majorities(self):
        # iterate through all rows
        majority_dataframe = self.dataframe.copy().drop(self.annotator_label_names + self.annotator_annotation_names, axis=1)
        majority_dataframe['annotated_by_how_many_annotators'] = 0
        for index, row in self.dataframe.iterrows():
            label = self.__calculate_majority_label(row[ANNOTATOR_LABEL_NAMES])
            
            # create Span objects for each annotator
            span = Span([], [], [], row['text'])
            if label != ANNOTATION_SKIPPED:
                for annotator_label_name, annotator_annotation_name in zip(self.annotator_label_names, self.annotator_annotation_names):
                    if (row[annotator_label_name] == label) and (row[annotator_label_name] != ANNOTATOR_NOT_SEEN):
                        span += MajorityData.encoded_span_string2span(self.duplicated_emojis, row['text'], row[annotator_annotation_name], label)
            encoded_majority_span_string = MajorityData.span2encoded_span_string(span)
            
            # create a new column
            row['annotated_by_how_many_annotators'] = (row[ANNOTATOR_LABEL_NAMES] != ANNOTATOR_NOT_SEEN).sum()
            
            # drop unnecessary columns
            row.drop(self.annotator_label_names + self.annotator_annotation_names, inplace=True)
            
            # edit some columns
            row['entities'] = encoded_majority_span_string
            row['label'] = label
            
            # add to majority dataframe
            majority_dataframe.loc[index] = row
        
        return majority_dataframe
    
    def __calculate_majority_label(self, annotator_labels):
        # see if skipped
        annotator_labels = annotator_labels.to_numpy()
        if ANNOTATOR_SKIPPED in annotator_labels: # skipped
            return ANNOTATION_SKIPPED
        
        # drop not seen to make it easier
        annotator_labels = annotator_labels[annotator_labels != ANNOTATOR_NOT_SEEN]
        
        # tie or majority
        unique_elements, counts = np.unique(annotator_labels, return_counts=True)
        max_count = np.max(counts)
        max_count_elements = unique_elements[counts == max_count]
        if len(max_count_elements) == 1:
            return max_count_elements[0]
        else:
            return -1
    
    @staticmethod
    def encoded_span_string2span(duplicated_emojis:bool, text:str, encoded_span_string:str, label:int):
        encoded_span_string_list = list(map(lambda x: x.split("%"), encoded_span_string.split("|")))
        text = text.replace("\r", "")
        intervals = []
        priorities = []
        labels = []
        
        if label != 0:
            if encoded_span_string:
                if duplicated_emojis:
                    duplicated_text = duplicate_emojis(text)
                else:
                    duplicated_text = text
                for encoded_span_string_loc, encoded_span_string_type in encoded_span_string_list:
                    encoded_span_string_loc = np.array(list(map(int,encoded_span_string_loc.split(":"))))
                    encoded_span_string_loc -= int(count_emojis(duplicated_text[:encoded_span_string_loc[0]])/2)
                    
                    intervals.append([encoded_span_string_loc[0], encoded_span_string_loc[1]])
                    priorities.append(PRIORITY_DICTIONARY[encoded_span_string_type])
                    labels.append(encoded_span_string_type)
                    
        else:
            return Span([], [], [])

        return Span(intervals, priorities, labels, text)
    
    @staticmethod
    def span2encoded_span_string(span:Span):
        encoded_span_string = ""
        for i, interval in enumerate(span.intervals):
            encoded_span_string += str(interval[0]) + ":" + str(interval[1]) + "%" + span.labels[i]
            if i != len(span.intervals) - 1:
                encoded_span_string += "|"
        return encoded_span_string
    
    @staticmethod
    def encoded_span_string_dataframe2span_list(encoded_span_string_dataframe:pd.DataFrame, duplicated_emojis:bool=True):
        span_list = []
        for index, row in encoded_span_string_dataframe.iterrows():
            span_list.append(MajorityData.encoded_span_string2span(duplicated_emojis, row['text'], row['entities'], row['label']))
        return span_list

class HateDataset(Dataset):
    def __init__(self, input_ids, attention_mask, sequence_labels, token_labels, tokenized_text, text, config):
        self.config = config
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.sequence_labels = sequence_labels
        self.token_labels = token_labels
        self.tokenized_text = tokenized_text
        self.text = text

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids':self.input_ids[idx], 
                'attention_mask':self.attention_mask[idx], 
                'sequence_label':self.sequence_labels[idx], 
                'token_labels':self.token_labels[idx], 
                'tokenized_text':self.tokenized_text[idx], 
                'text':self.text[idx]}
    
    def shuffle(self):
        permutation = torch.randperm(len(self.input_ids))
        self.input_ids = self.input_ids[permutation]
        self.attention_mask = self.attention_mask[permutation]
        self.sequence_labels = self.sequence_labels[permutation]
        self.token_labels = self.token_labels[permutation]
        self.tokenized_text = self.tokenized_text[permutation]
        self.text = self.text[permutation]

    def split(self, split_ratio:float, k:int):
        max_k = int(1/(1-split_ratio))
        if (k > max_k) or (k < 0): raise ValueError(f'k must be less than {max_k}')
        slice_len = int(len(self) * (1-split_ratio))
        dataset_eval = HateDataset(self.input_ids[k*slice_len:(k+1)*slice_len], 
                                   self.attention_mask[k*slice_len:(k+1)*slice_len], 
                                   self.sequence_labels[k*slice_len:(k+1)*slice_len], 
                                   self.token_labels[k*slice_len:(k+1)*slice_len], 
                                   self.tokenized_text[k*slice_len:(k+1)*slice_len], 
                                   self.text[k*slice_len:(k+1)*slice_len],
                                   self.config)
        dataset_train = HateDataset(torch.cat((self.input_ids[:k*slice_len], self.input_ids[(k+1)*slice_len:])),
                                    torch.cat((self.attention_mask[:k*slice_len], self.attention_mask[(k+1)*slice_len:])),
                                    torch.cat((self.sequence_labels[:k*slice_len], self.sequence_labels[(k+1)*slice_len:])),
                                    torch.cat((self.token_labels[:k*slice_len], self.token_labels[(k+1)*slice_len:])),
                                    np.concatenate((self.tokenized_text[:k*slice_len], self.tokenized_text[(k+1)*slice_len:])),
                                    np.concatenate((self.text[:k*slice_len], self.text[(k+1)*slice_len:])),
                                    self.config)
        return dataset_train, dataset_eval

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.sequence_labels = self.sequence_labels.to(device)
        self.token_labels = self.token_labels.to(device)
        return self

class JointModel(nn.Module):
    def __init__(self, language:Literal['tr', 'en'], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.number_of_labels = len(LABEL_DICTIONARY) - 2 # 2 for skipped and tie
        self.number_of_tokens = len(TOKEN_DICTIONARY)*2-1 if TAG_TYPE=='BIO' else len(TOKEN_DICTIONARY) # 2 for B and I, 1 for O
        if language == 'tr': self.bert = transformers.AutoModel.from_pretrained(TURKISH_MODEL)
        elif language == 'en': self.bert = transformers.AutoModel.from_pretrained(ENGLISH_MODEL)
        else: raise ValueError('language must be either turkish or english')
        self.dropout = nn.Dropout(0.1)
        self.sequence_classifier = nn.Linear(768, self.number_of_labels)
        self.token_classifier = nn.Linear(768, self.number_of_tokens)
        self.sequence_loss = nn.CrossEntropyLoss( )
        self.token_loss = nn.CrossEntropyLoss( )
        self.softmax = nn.Softmax(dim=1)
        self.token_softmax = nn.Softmax(dim=2)
        self.token_loss_weight = 0.9
    
    def forward(self, input_ids, attention_mask, sequence_label, token_labels, *args, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        if 'pooler_output' in outputs.keys():
            pooler_output = outputs.pooler_output
        else:
            pooler_output = last_hidden_state[:, 0, :]
        
        sequence_logits = self.sequence_classifier(self.dropout(pooler_output))
        token_logits = self.token_classifier(self.dropout(last_hidden_state))
        
        seq_loss = self.sequence_loss(sequence_logits, sequence_label)
        tok_loss = self.token_loss(token_logits.view(-1, self.number_of_tokens), token_labels.view(-1))
        loss = seq_loss * (1-self.token_loss_weight) + self.token_loss_weight * tok_loss
        
        return {'loss':loss, 'sequence_logits':sequence_logits, 'token_logits':token_logits}
    
    def predict(self, input_ids, attention_mask, *args, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_logits = self.sequence_classifier(self.dropout(outputs.pooler_output))
        token_logits = self.token_classifier(self.dropout(outputs.last_hidden_state))
        sequence_predictions = torch.argmax(self.softmax(sequence_logits), dim=1)
        token_predictions = torch.argmax(self.token_softmax(token_logits), dim=2)
        return {'sequence_predictions':sequence_predictions, 'token_predictions':token_predictions}

def compute_metrics(eval_pred):
    # input
    predictions, labels = eval_pred
    sequence_predictions, token_predictions = predictions
    sequence_label, token_label = labels
    
    # to pytorch
    sequence_predictions = torch.tensor(sequence_predictions)
    sequence_label = torch.tensor(sequence_label)
    token_predictions = torch.tensor(token_predictions)
    token_label = torch.tensor(token_label)
    
    number_of_sequence_labels = sequence_predictions.shape[-1]
    number_of_token_labels = token_predictions.shape[-1]
    # sequence_predictions shape: (batch_size, number_of_sequence_labels)
    # sequence_label shape: (batch_size,)
    # token_predictions shape: (batch_size, max_seq_len, number_of_token_labels)
    # token_label shape: (batch_size, max_seq_len)
    token_predictions = token_predictions.view(-1, token_predictions.shape[-1])
    token_label = token_label.view(-1)
    # token_predictions shape: (batch_size*max_seq_len, number_of_token_labels)
    # token_label shape: (batch_size*max_seq_len,)
    
    # ignore IGNORE_INDEX
    #sequence_predictions = sequence_predictions[sequence_label != IGNORE_INDEX]
    #sequence_label = sequence_label[sequence_label != IGNORE_INDEX]
    #token_predictions = token_predictions[token_label != IGNORE_INDEX]
    #token_label = token_label[token_label != IGNORE_INDEX]
    
    # make predictions
    sequence_predictions = torch.max(sequence_predictions, dim=1)[1]
    token_predictions = torch.max(token_predictions, dim=1)[1]
    # sequence_predictions shape: (batch_size,)
    # sequence_label shape: (batch_size,)
    # token_predictions shape: (batch_size*max_seq_len,)
    # token_label shape: (batch_size*max_seq_len,)
    
    # f1 for each sequence and token label
    f1_score_sequence = MulticlassF1Score(average=None, num_classes=number_of_sequence_labels, ignore_index=IGNORE_INDEX)(sequence_predictions, sequence_label)
    f1_score_token = MulticlassF1Score(average=None, num_classes=number_of_token_labels, ignore_index=IGNORE_INDEX)(token_predictions, token_label)
    
    # calculate macro f1
    f1_score_sequence_macro = f1_score_sequence[f1_score_sequence != None]
    f1_score_sequence_macro = f1_score_sequence_macro.mean().item()
    f1_score_token_macro = f1_score_token[f1_score_token != None]
    f1_score_token_macro = f1_score_token_macro.mean().item()
    
    return_dict = {'f1_score_sequence_macro':f1_score_sequence_macro, 'f1_score_token_macro':f1_score_token_macro}
    for i in range(number_of_sequence_labels):
        return_dict['f1_score_sequence_' + str(i)] = f1_score_sequence[i]
    for i in range(number_of_token_labels):
        return_dict['f1_score_token_' + str(i)] = f1_score_token[i]
    
    # return
    return return_dict 

