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
train = pd.read_csv(path + "subtask-2/modified-train.csv")
# test = pd.read_csv("subtask-1/modified-test.csv")
test = pd.read_csv(path + "subtask-2/modified-test.csv")

class Subtask1_Model(tf.keras.Model):
    def __init__(self):
        super(Subtask1_Model, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        # self.dense_layer3 = tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32')
        self.dense_layer2= tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dense_layer2= tf.keras.layers.Dense(128, activation="sigmoid",dtype='float32')
        self.dense_layer1 = tf.keras.layers.Dense(1, activation="sigmoid",dtype='float32')
        self.dropout = tf.keras.layers.Dropout(0.4)

    def __call__(self, x_train):
        intermediate = self.bert_layer(x_train)
#         intermediate2 = self.dropout(intermediate[1])
#         output1 = self.dense_layer2(intermediate2)
#         output2 = self.dense_layer1(output1)
        intermediate2 = self.dense_layer2(intermediate[1])
        output1 = self.dropout(intermediate2)
        output2 = self.dense_layer1(output1)
        
        return output2
    
        


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


def generateIndicator(o1, o2, gT):
    o1 = o1*3
    o2 = o2*3
    label = []
    for x,y in zip(o1,o2):
        if(x-y<-0.000001):
            label.append(2)
        elif(x-y>0.000001):
            label.append(1)
        else:
            label.append(0)
    
    indicator = []
    for x,y in zip(gT, label):
        if(x==y):
            indicator.append(-1.0)
        else:
            indicator.append(1.0)
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
    target1 = []
    target2 = []
    labels = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids1.append(in_id1)
        attn_mask1.append(attn1)
        segment_ids1.append(seg1)
        target1.append(row["meanGrade1"]/3)

        in_id1 = tokenizer.encode(row['original'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids2.append(in_id1)
        attn_mask2.append(attn1)
        segment_ids2.append(seg1)
        target2.append(row["meanGrade2"]/3)

        labels.append(row["label"])

    input_dictionary1 = {}
    input_dictionary2 = {}

    with tf.GradientTape() as tape:
        input_dictionary1['input_ids'] = tf.convert_to_tensor(np.array(input_ids1))
        input_dictionary1['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask1))
        input_dictionary1['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids1))
        
        input_dictionary2['input_ids'] = tf.convert_to_tensor(np.array(input_ids2))
        input_dictionary2['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask2))
        input_dictionary2['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids2))
        
        output1 = model(input_dictionary1)
        output2 = model(input_dictionary2)

        target1 = tf.convert_to_tensor(np.array(target1), dtype="float32")
        target1 = tf.reshape(target1 ,shape=(target1.shape[0],1))
        target2 = tf.convert_to_tensor(np.array(target2), dtype="float32")
        target2 = tf.reshape(target2 ,shape=(target2.shape[0],1))
        
        loss = getReward(output1, output2, labels)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
    return loss

# def testing(test_data):
#     loss = 0
#     input_dictionary = {}
    
#     for index, row in test_data.iterrows():
#         in_id1 = tokenizer.encode(row['original'], row["edit"], add_special_tokens=True)
#         attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
#         in_id1 += [0]*(maxLengthPadding-len(in_id1))

#         input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
#         input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
#         input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
#         output = model(input_dictionary)
#         loss+= (output*3-row["meanGrade"])**2
#         if(index%500==0):
#             print(loss)
#     return math.sqrt(loss/test_data.shape[0])

def testing(test_data):
    loss = 0
    input_dictionary = {}
    correct = 0
    targets = []
    o1 = []
    o2 = []
    lbls = []
    for index, row in test_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        # targets = [row["meanGrade1"], row["meanGrade2"]]

        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
        output1 = model(input_dictionary)
        o1.append(output1)
        output1 = output1*3
        
        
        
        
        in_id1 = tokenizer.encode(row['original'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        # targets = [row["meanGrade1"], row["meanGrade2"]]

        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
        output2 = model(input_dictionary)
        o2.append(output2)
        output2 = output2*3
        
        # print(output)
        # print(output.shape)
        # for x in output:
        #     currentVal = 0
        #     if(x<0.66):
        #         currentVal = 0
        #     elif(x>1.33):
        #         currentVal = 2
        #     else:
        #         currentVal = 1  
        
        if(output1-output2<-0.000001):
            output = 2
        elif(output1-output2>0.000001):
            output = 1
        else:
            output = 0  
        
        lbls.append(row['label'])
        if(output==row['label']):
            correct+=1

        if(index < 50):
          print(output1.numpy(), output2.numpy(), row['label'], output)

        if(index>0 and index%500==0):
            print("Accuracy: ", float(correct/index))
            print("Reward: ", -1*getReward(o1, o2, lbls))
    return float(correct/test_data.shape[0])


# training()

batch_size = 16
num_batches = math.floor(train.shape[0]/batch_size)
print(num_batches)
EPOCHS = 10

for e in range(10):
    loss = 0
    for i in range(num_batches-1):
        index = i*batch_size
        curr_batch = train[index:index+batch_size]
        loss = training(curr_batch)
        print("Loss after step", i ,"is:", loss)

    curr_batch = train[num_batches*batch_size:]
    if(curr_batch.shape[0]>0):
        loss = training(curr_batch)
    print("Loss after epoch is:", loss)

## Testing 

    loss = testing(test)
    print("Mean Squared Error Loss is :" , loss)