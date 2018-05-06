import os
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse

import model
import evaluation

# online
PATH_TO_TRAIN = '/content/data/rsc15_train_tr.txt'
PATH_TO_TEST = '/content/data/rsc15_test.txt'


# data = pd.read_csv(PATH_TO_TRAIN, sep='\t', dtype={'ItemId': np.int64})
# valid = pd.read_csv(PATH_TO_TEST, sep='\t', dtype={'ItemId': np.int64})


class DataUtils(object):
    def __init__(self, data, valid=None, args=None):
        self.data = data
        self.valid = valid

        self.session_key = args.session_key
        self.item_key = args.item_key
        self.time_key = args.time_key

        self.n_items = 0
        self.itemidmap = {}
        self.lived_t = []

        self.n_days = 0

        def add_itemIdx():
            """
            add 0-N index for item ('ItemIdx')
            """
            itemids = self.data[self.item_key].unique()
            self.n_items = len(itemids)
            self.itemidmap = pd.DataFrame({self.item_key: itemids, 'ItemIdx': np.arange(self.n_items)})
            self.data = pd.merge(self.data, self.itemidmap, on=self.item_key, how='inner')

        add_itemIdx()

    def item_lived_time(self):
        """
        compute item lived time statistics
        """
        data = self.data.copy()
        grouped_item = data.groupby('ItemIdx', as_index=False)[self.time_key]
        # item group by id, selected by time ['ItemIdx', 'Time']

        min_t = grouped_item.agg(np.min)
        min_t.columns = ['ItemIdx', 'Min_t']  # rename column
        x = pd.merge(data, min_t, on='ItemIdx')  # merge min_t to data
        x['Lived_t'] = x[self.time_key] - x['Min_t']
        lived_t = x['Lived_t']  # Series

        self.lived_t = lived_t / 86400
        print(self.lived_t.describe())

    def session_stat(self):
        """
        1.num of days
        2.num of session in every day
        3.session length distribution
        """
        # Full train set
        # 	Events: 31637239
        # 	Sessions: 7966257
        # 	Items: 37483
        # Test set
        # 	Events: 71222
        # 	Sessions: 15324
        # 	Items: 6751
        # Train set
        # 	Events: 31579006
        # 	Sessions: 7953885
        # 	Items: 37483

        # 1.num of days
        self.n_days = (self.data[self.time_key].max() - self.data[self.time_key].min()) / 86400
        print('n_days: %s' % self.n_days)

        # 2.session in every day and 3.session length distribution
        data = self.data.copy()
        data[self.time_key] -= data[self.time_key].min()
        data[self.time_key] /= 86400
        sess_t = data.groupby(self.session_key, as_index=False)[self.time_key].agg(np.min)  # ['session_id', 'Time']
        sess_stat = sess_t.groupby(self.time_key).size()  # num of session in every day  #  ['Time', size]
        print('Session stat: ')
        print(sess_stat.describe())





