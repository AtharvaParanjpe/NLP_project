import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle

from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification

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



dataLoader = DataLoader()















