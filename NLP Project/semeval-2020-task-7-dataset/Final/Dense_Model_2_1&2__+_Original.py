from google.colab import drive
import pandas as pd
import numpy as np
from scipy import sparse
# import os
from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf
import math


drive.mount('/content/drive', force_remount=True)

path = "/content/drive/My Drive/NLP Project/NLP_project/NLP Project/semeval-2020-task-7-dataset/"

dev = pd.read_csv(path + "subtask-1/modified-dev.csv")
train2 = pd.read_csv(path + "subtask-1/combined-train.csv")
train2 = train2.dropna()
train = pd.read_csv(path + "subtask-2/modified-train.csv")
# test = pd.read_csv("subtask-1/modified-test.csv")
test = pd.read_csv(path + "subtask-2/modified-test.csv")

class Subtask1_Model(tf.keras.Model):
    def __init__(self):
        super(Subtask1_Model, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense_layer4 = tf.keras.layers.Dense(512, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001)) 
        self.dense_layer3= tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_layer2 = tf.keras.layers.Dense(128, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dense_layer1 = tf.keras.layers.Dense(1, activation="sigmoid",dtype='float32')
        self.dropout = tf.keras.layers.Dropout(0.5)

    def __call__(self, x_train):
        intermediate = self.bert_layer(x_train)
        intermediate2 = self.dense_layer4(intermediate[1])

        output1 = self.dropout(intermediate2)
        output2 = self.dense_layer3(output1)

        output3 = self.dropout(output2)
        output4 = self.dense_layer2(output3)

        output5 = self.dropout(output4)
        output6 = self.dense_layer1(output5)
        return output6
    
maxLengthPadding = 80
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = Subtask1_Model()
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

def training(train_data):
    loss = 0
    input_ids = [] 
    attn_mask = []
    segment_ids = []
    targets = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids.append(in_id1)
        attn_mask.append(attn1)
        segment_ids.append(seg1)
        targets.append(row["meanGrade1"]/3)
    input_dictionary = {}

    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids))
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))
        output = model(input_dictionary)

        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        targets = tf.reshape(targets ,shape=(targets.shape[0],1))
        
        result = tf.math.subtract(targets,output)
        squaredError = tf.math.square(result)
        loss = tf.math.divide(tf.math.reduce_sum(squaredError),squaredError.shape[0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss = 0
    input_ids = [] 
    attn_mask = []
    segment_ids = []
    targets = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["original"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids.append(in_id1)
        attn_mask.append(attn1)
        segment_ids.append(seg1)
        targets.append(0)
        
    input_dictionary = {}

    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids))
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))
        output = model(input_dictionary)

        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        targets = tf.reshape(targets ,shape=(targets.shape[0],1))
        
        result = tf.math.subtract(targets,output)
        squaredError = tf.math.square(result)
        loss = tf.math.divide(tf.math.reduce_sum(squaredError),squaredError.shape[0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    loss = 0
    input_ids = [] 
    attn_mask = []
    segment_ids = []
    targets = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids.append(in_id1)
        attn_mask.append(attn1)
        segment_ids.append(seg1)
        targets.append(row["meanGrade2"]/3)
        
    input_dictionary = {}

    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids))
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))
        output = model(input_dictionary)

        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        targets = tf.reshape(targets ,shape=(targets.shape[0],1))
        
        result = tf.math.subtract(targets,output)
        squaredError = tf.math.square(result)
        loss = tf.math.divide(tf.math.reduce_sum(squaredError),squaredError.shape[0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss = 0
    input_ids = [] 
    attn_mask = []
    segment_ids = []
    targets = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['edit1'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids.append(in_id1)
        attn_mask.append(attn1)
        segment_ids.append(seg1)
        targets.append((row["meanGrade2"]- row["meanGrade1"])/3)
    input_dictionary = {}

    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids))
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))
        output = model(input_dictionary)

        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        targets = tf.reshape(targets ,shape=(targets.shape[0],1))
        
        result = tf.math.subtract(targets,output)
        squaredError = tf.math.square(result)
        loss = tf.math.divide(tf.math.reduce_sum(squaredError),squaredError.shape[0])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
    

def testing(test_data):
    loss = 0
    input_dictionary = {}
    correct = 0
    targets = []
    for index, row in test_data.iterrows():
        in_id1 = tokenizer.encode(row['edit1'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
        output = model(input_dictionary)
        output = output*3
        
        if(index<50):
          print(row["label"], output)


        if(output<0.4999):
            output = 1
        elif(output>0.5001):
            output = 2
        else:
            output = 0  
        
        if(output==row['label']):
            correct+=1

        if(index>0 and index%100==0):
            # file.write("Accuracy after " + str(index) + " is: " + str(float(correct/index)) + "\n")
            print("Accuracy: ", float(correct/index))
    return float(correct/test_data.shape[0])

batch_size = 16
num_batches = math.floor(train.shape[0]/batch_size)
num_batches2 = math.floor(train2.shape[0]/batch_size)
print(num_batches)
EPOCHS = 10
# file = open("./MyFile.txt","w")

for e in range(10):
    loss = 0
    for i in range(num_batches - 1):
        index = i*batch_size
        curr_batch = train[index:index+batch_size]
        loss = training(curr_batch)
        print("Loss after step", i ,"is:", loss)
      
    for j in range(num_batches2 -1):
        index = j*batch_size
        curr_batch = train2[index:index+batch_size]
        loss = training(curr_batch)
        print("Loss after step", j ,"is:", loss)

    curr_batch = train[num_batches*batch_size:]
    if(curr_batch.shape[0]>0):
        loss = training(curr_batch)
    print("Loss after epoch is:", loss)

## Testing 

    loss = testing(test)
    print("Mean Squared Error Loss is :" , loss)
    
    # file.close()
    # break