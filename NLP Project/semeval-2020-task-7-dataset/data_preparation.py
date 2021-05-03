import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification

from itertools import combinations

class DataLoader():

    def __init__(self):
        self.dev1 = pd.read_csv("subtask-1/dev.csv")
        self.train1 = pd.read_csv("subtask-1/train.csv")
        self.test1 = pd.read_csv("subtask-1/test.csv")

        self.dev2 = pd.read_csv("subtask-2/dev.csv")
        self.train2 = pd.read_csv("subtask-2/train.csv")
        self.test2 = pd.read_csv("subtask-2/test.csv")

        self.clean_data_for_tasks()
    
    def replaceEditForTask1(self, file):
        file = file.drop(["id", "grades"], axis=1)
        for index, row in file.iterrows():
            ## split at < and >  and replace the word with edited value
            split = row["original"].split("<")
            
            ## make it original w/o <> and edit as the edited headline
            split[1] = split[1].split(">")
            split[1][0] = split[1][0][:-1]
            original = [split[0], split[1][0], split[1][1]]
            original = "".join(original)
            edited = [split[0], row["edit"], split[1][1]]
            edited = "".join(edited)
            file.loc[index, "edit"] = edited
            file.loc[index, "original"] = original
        return file

    def replaceEditForTask2(self, file):
        file = file.drop(["id", "grades1", 'grades2'], axis=1)
        for index, row in file.iterrows():
            ## split at < and >  and replace the word with edited value
            split = row["original1"].split("<")
            split[1] = split[1].split(">")
            split[1][0] = split[1][0][:-1]
            original = [split[0], split[1][0], split[1][1]]
            original = "".join(original)
            edited = [split[0], row["edit1"], split[1][1]]
            edited = "".join(edited)
            file.loc[index, "edit1"] = edited
            file.loc[index, "original1"] = original

            split = row["original2"].split("<")
            split[1] = split[1].split(">")
            split[1][0] = split[1][0][:-1]
            original = [split[0], split[1][0], split[1][1]]
            original = "".join(original)
            edited = [split[0], row["edit2"], split[1][1]]
            edited = "".join(edited)
            file.loc[index, "edit2"] = edited
            file.loc[index, "original2"] = original
            
            file.loc[index, "original"] = original
            
        return file

    def clean_data_for_tasks(self):
        # self.train1 = self.replaceEditForTask1(self.train1)
        # self.test1 = self.replaceEditForTask1(self.test1)
        # self.dev1 = self.replaceEditForTask1(self.dev1)

        # self.train1.to_csv("./subtask-1/modified-train.csv")
        # self.test1.to_csv("./subtask-1/modified-test.csv")
        # self.dev1.to_csv("./subtask-1/modified-dev.csv")

        self.train2 = self.replaceEditForTask2(self.train2)
        self.test2 = self.replaceEditForTask2(self.test2)
        self.dev2 = self.replaceEditForTask2(self.dev2)

        self.train2.to_csv("./subtask-2/modified-train.csv")
        self.test2.to_csv("./subtask-2/modified-test.csv")
        self.dev2.to_csv("./subtask-2/modified-dev.csv")

    def makeCombinationOfEdits(self):
        #df = pd.read_csv('subtask-1/modified-train.csv')
        df = pd.read_csv('subtask-1/modified-test.csv')

        final_df = pd.DataFrame()
        no_combine_df = pd.DataFrame()
        
        gk = df.groupby('original')
        
        for gname in gk.groups.keys():
            grouped_df = gk.get_group(gname)
            
            if(grouped_df.shape[0] > 1):
                a, b = map(list, zip(*combinations(grouped_df.index, 2)))

                temp_df = pd.concat([grouped_df.loc[a].reset_index(drop = True), 
                            grouped_df.loc[b].reset_index(drop = True)], keys = ['a', 'b'], axis=1)

                temp_df.drop(temp_df.columns[[0, 4, 5]], axis=1, inplace=True)

                final_df = pd.concat([final_df.reset_index(drop=True), temp_df.reset_index(drop=True)])

            else:
                grouped_df.drop(grouped_df.columns[[0]], axis=1, inplace=True)
                grouped_df["edit2"] = ""
                grouped_df["meanGrade2"] = ""
                no_combine_df = pd.concat([no_combine_df.reset_index(drop=True), grouped_df.reset_index(drop=True)])

        final_df.columns = ['original', 'edit1', 'meanGrade1', 'edit2', 'meanGrade2']
        no_combine_df.columns = ['original', 'edit1', 'meanGrade1', 'edit2', 'meanGrade2']

        final_df = pd.concat([final_df.reset_index(drop=True), no_combine_df.reset_index(drop=True)])

        print(final_df.info())
            
        final_df.to_csv('subtask-1/combined-test.csv', index=False)



dataLoader = DataLoader()
dataLoader.makeCombinationOfEdits()















