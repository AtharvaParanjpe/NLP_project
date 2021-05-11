from google.colab import drive
import pandas as pd
import numpy as np
from scipy import sparse
# import os
import sklearn
from sklearn.metrics import confusion_matrix

from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf
import math

drive.mount('/content/drive', force_remount=True)

path = "/content/drive/My Drive/NLP Project/NLP_project/NLP Project/semeval-2020-task-7-dataset/"
dev = pd.read_csv(path + "subtask-2/modified-dev.csv")
train = pd.read_csv(path + "subtask-2/modified-train.csv")
# test = pd.read_csv("subtask-1/modified-test.csv")
test = pd.read_csv(path + "subtask-2/modified-test.csv")

class Subtask2_Model(tf.keras.Model):
    def __init__(self):
        super(Subtask2_Model, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense_layer4 = tf.keras.layers.Dense(512, activation="tanh",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
        self.dense_layer3= tf.keras.layers.Dense(256, activation="tanh",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_layer2 = tf.keras.layers.Dense(128, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_layer1 = tf.keras.layers.Dense(1, activation="sigmoid",dtype='float32')
        self.dropout = tf.keras.layers.Dropout(0.4)

    def __call__(self, x_train1, x_train2):
        in1 = self.bert_layer(x_train1)
        in2 = self.bert_layer(x_train2)
        temp = tf.concat([in1[1], in2[1], tf.abs(tf.math.subtract(in1[1], in2[1])), tf.math.multiply(in1[1], in2[1])], 1)
        intermediate = self.dense_layer4(temp)

        output1 = self.dropout(intermediate)
        output2 = self.dense_layer3(output1)

        output3 = self.dropout(output2)
        output4 = self.dense_layer2(output3)

        output5 = self.dropout(output4)
        output6 = self.dense_layer1(output5)
        
        return output6

maxLengthPadding = 80

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = Subtask2_Model()
optimizer = tf.keras.optimizers.Adam( learning_rate=0.00001)

def get_bert_params(tokens):
    attn_mask = []
    seg_ids = []
    if(len(tokens)<maxLengthPadding):
        attn_mask = [1]*len(tokens) + [0]*(maxLengthPadding-len(tokens))
    else:
        attn_mask = [1]*maxLengthPadding
    segment = 0
    for x in tokens:
        seg_ids.append(segment)
        if(x=='[SEP]'):
            segment = 1
    seg_ids+=[0]*(maxLengthPadding-len(tokens))
    return attn_mask,seg_ids

def generateIndicator(o1, o2, gT):
    o1 = o1*3
    o2 = o2*3
    label = []
    for x,y in zip(o1,o2):
        if(x-y<-0.0001):
            label.append(2)
        elif(x-y>0.0001):
            label.append(1)
        else:
            label.append(0)
    
    indicator = []
    for x,y in zip(gT, label):
        if(x==y):
            indicator.append(-1)
        else:
            indicator.append(1)
    return tf.convert_to_tensor(np.array(indicator), dtype="float32")

def getReward(output1, output2, labels):
    indicator = generateIndicator(output1, output2, labels)
    indicator = tf.reshape(indicator ,shape=(indicator.shape[0],1))
    absError = tf.math.abs(tf.subtract(output1*3, output2*3))
    error = tf.math.multiply(indicator, absError)
    loss = tf.math.reduce_mean(error)
    return loss

def training(train_data):
    loss = 0
    input_ids1 = [] 
    attn_mask1 = []
    segment_ids1 = []
    input_ids2 = [] 
    attn_mask2 = []
    segment_ids2 = []
    input_ids3 = [] 
    attn_mask3 = []
    segment_ids3 = []
    
    targets = []
    targets1 = []
    targets2 = []
    labels = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(["original"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids1.append(in_id1)
        attn_mask1.append(attn1)
        segment_ids1.append(seg1)
        
        in_id1 = tokenizer.encode(row['edit1'], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids2.append(in_id1)
        attn_mask2.append(attn1)
        segment_ids2.append(seg1)
        
        in_id1 = tokenizer.encode(row['edit2'], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids3.append(in_id1)
        attn_mask3.append(attn1)
        segment_ids3.append(seg1)
        
        targets1.append(row["meanGrade1"]/3)
        targets2.append(row["meanGrade2"]/3)
# 
        # targets.append([row["meanGrade1"]/3, row["meanGrade2"]/3])
        labels.append(row["label"])
        
    input_dictionary1 = {}
    input_dictionary2 = {}
    input_dictionary3 = {}

    with tf.GradientTape() as tape:
        input_dictionary1['input_ids'] = tf.convert_to_tensor(np.array(input_ids1))
        input_dictionary1['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask1))
        input_dictionary1['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids1))
        
        input_dictionary2['input_ids'] = tf.convert_to_tensor(np.array(input_ids2))
        input_dictionary2['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask2))
        input_dictionary2['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids2))
        
        input_dictionary3['input_ids'] = tf.convert_to_tensor(np.array(input_ids3))
        input_dictionary3['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask3))
        input_dictionary3['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids3))
        
        output1 = model(input_dictionary1, input_dictionary2)
        output2 = model(input_dictionary1, input_dictionary3)
        
#         output = tf.concat([output1, output2], axis = 1)
#         targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        targets1 = tf.convert_to_tensor(np.array(targets1), dtype="float32")
        targets1 = tf.reshape(targets1 ,shape=(targets1.shape[0],1))
        targets2 = tf.convert_to_tensor(np.array(targets2), dtype="float32")
        targets2 = tf.reshape(targets2 ,shape=(targets2.shape[0],1))
        indicator = generateIndicator(output1, output2, labels)
        indicator = tf.reshape(indicator ,shape=(indicator.shape[0],1))
      
        # result1 = tf.math.subtract(targets1,output1)
        # result2 = tf.math.subtract(targets2,output2)
        # absError = tf.math.abs(result1+result2)
        absError = tf.subtract(output1*3, output2*3)

        error = tf.math.multiply(indicator, tf.math.abs(absError))
        # print(indicator.shape)
        # print(absError.shape)
        # print(error.shape)
        # input()
        
        
        loss = tf.math.reduce_mean(error)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss



def testing(test_data):
    loss = 0
    input_dictionary1 = {}
    input_dictionary2 = {}
    input_dictionary3 = {}
    correct = 0
    o1 = []
    o2 = []
    labels = []
    pred = []
    targets = []
    for index, row in test_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        input_dictionary1['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary1['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary1['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]

        in_id1 = tokenizer.encode(row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        input_dictionary2['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary2['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary2['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
                
        in_id1 = tokenizer.encode(row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        input_dictionary3['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary3['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary3['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]

        output1 = model(input_dictionary1, input_dictionary2)
        output2 = model(input_dictionary1, input_dictionary3)
        
        o1.append(output1)
        o2.append(output2)

        output1 = output1*3
        output2 = output2*3

        labels.append(row["label"])

        if(index<50):
          print(output1, output2, row["label"])
        if(output1-output2<-0.0001):
            output = 2
        elif(output1-output2>0.0001):
            output = 1
        else:
            output = 0  
        pred.append(output)
        if(output==row['label']):
            correct+=1

        if(index>0 and index%500==0):
            print("Accuracy: ", float(correct/index))
            print("Reward: ", -1*getReward(o1, o2, labels))
            print("Confusion Matrix : ", confusion_matrix(labels, pred))
    return float(correct/test_data.shape[0])


# training()

batch_size = 16
num_batches = math.floor(train.shape[0]/batch_size)
print(num_batches)
EPOCHS = 10

for e in range(10):
    loss = 0
    for i in range(num_batches - 1):
        index = i*batch_size
        curr_batch = train[index:index+batch_size]
        loss = training(curr_batch)
        print("Loss after step", i ,"is:", loss)

    curr_batch = train[num_batches*batch_size:]
    if(curr_batch.shape[0]>0):
        loss = training(curr_batch)
    print("Loss after epoch is:", loss)

## Testing 

    accuracy = testing(test)
    print("Final Accuracy :" , accuracy)
