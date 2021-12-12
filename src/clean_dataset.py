import pandas as pd

def prune_attr(data):
    correlation = data.corr()['results'].to_frame().T
    thresh = 0.3
    attr = list(filter(lambda x: abs(float(correlation[x])) > thresh, data.columns))
    print('Total of useful attributes:', len(attr), '.')
    #results will appear in this list but of course it won't be there in the test data
    pruned_df = data[attr]

    return pruned_df