from torch.utils.data import Dataset
import pandas as pd

try:
    from read_data import read_annotated_data, read_initial_data
    from commons import *
except ModuleNotFoundError: 
    from source.read_data import read_annotated_data, read_initial_data
    from source.commons import *

def test_span():
    span1 = Span([(0, 2), (4, 6), (10, 15), (16, 17)], [1, 2, 3, 4], ['a', 'b', 'c', 'd'])
    span2 = Span([(1, 3), (5, 7), (11, 12), (18, 19)], [3, 2, 3, 4], ['c', 'b', 'c', 'd'])
    span3 = span1 + span2
    print(span3) # Span([(16, 17), (18, 19), (10, 15), (1, 3), (4, 7)], [4, 4, 3, 3, 2], ['d', 'd', 'c', 'c', 'b'])
    
    span1 = Span([(0, 1), (2, 3)], [1, 3], ['a', 'c'])
    span2 = Span([(1, 2)], [2], ['b'])
    span3 = span1 + span2
    print(span3) # Span([(0, 1), (2, 3)], [1, 3], ['a', 'c'])
    
    span1 = Span([(1, 2)], [2], ['b'])
    span2 = Span([], [], [])
    span3 = span1 + span2
    print(span3) # Span([(1, 2)], [2], ['b'])

def write_majority_annotated_data():
    dataframe = read_annotated_data()
    majority_data = MajorityData(dataframe)
    df = majority_data.calculate_majorities()
    df.to_csv('data/majority_annotated_data/majority_annotated_data.tsv', sep='\t', index=True, header=True)

def write_combined_data():
    df1 = read_annotated_data()
    majority_data = MajorityData(df1)
    df1 = majority_data.calculate_majorities()
    df2 = read_initial_data()
    df2.rename(columns={'label': 'old_label'}, inplace=True)
    df2.drop(['language', 'text'], axis=1, inplace=True)
    df = pd.concat([df1, df2], axis=1)
    df.to_csv('data/combined_data/combined_data.tsv', sep='\t', index=True, header=True)

if __name__ == '__main__':
    # test Span
    test_span()
    
    # write out majority annotated data
    write_majority_annotated_data()
    
    # write out combined data
    read_annotated_data()