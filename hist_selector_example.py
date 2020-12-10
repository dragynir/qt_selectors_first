from os.path import dirname, join
import pandas as pd

import sys 
sys.path.append('..')
from plot_utils import hist_cluster_selector


df_all = pd.read_csv(join(dirname(__file__), '../data/merged_data.csv'))
map_cols = list(df_all.columns[3:])

hist_cluster_selector(df=df_all, cols=map_cols, mode='server')
