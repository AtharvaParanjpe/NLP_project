from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf
import math


dev = pd.read_csv("subtask-2/modified-dev.csv")
train = pd.read_csv("subtask-2/modified-train.csv")
test = pd.read_csv("subtask-2/modified-test.csv")


class Subtask2_Model(tf.keras.Model):
    def __init__(self):
        super(Subtask2_Model, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        self.dense_layer2= tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32')
        self.dense_layer1 = tf.keras.layers.Dense(2, activation="sigmoid",dtype='float32')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def __call__(self, x_train):
        intermediate = self.bert_layer(x_train)
        intermediate2 = self.dropout(intermediate[1])
        output1 = self.dense_layer2(intermediate2)
        output2 = self.dense_layer1(output1)
        return output2

maxLengthPadding = 160

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = Subtask2_Model()
optimizer = tf.keras.optimizers.Adam( learning_rate=0.001)

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
    labels = []


    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'] + ' ' + row['edit1'], row['original'] + ' ' + row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids.append(in_id1)
        attn_mask.append(attn1)
        segment_ids.append(seg1)
        targets.append([row["meanGrade1"]/3, row["meanGrade2"]/3])
        labels.append(row["label"]/2)
        
    input_dictionary = {}

    with tf.GradientTape() as tape:
        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(input_ids))
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask))
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids))
        output = model(input_dictionary)

        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")
        # targets = tf.reshape(targets ,shape=(targets.shape[0],1))
        
        labels = tf.convert_to_tensor(np.array(labels), dtype="float32")
        labels = tf.reshape(labels ,shape=(labels.shape[0],1))

        # beforeThreshold = tf.math.subtract(targets,output)
        
        result = tf.math.subtract(targets,output)
        squaredError = tf.math.abs(result)
        
        loss = tf.math.reduce_mean(tf.math.reduce_sum(squaredError, axis=1))#,squaredError.shape[0])
    
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def testing(test_data):
    loss = 0
    input_dictionary = {}
    correct = 0
    targets = []
    for index, row in test_data.iterrows():
        in_id1 = tokenizer.encode(row['original'] + ' ' + row['edit1'], row['original'] + ' ' + row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        # targets = [row["meanGrade1"], row["meanGrade2"]]

        input_dictionary['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
        output = model(input_dictionary)
        output = output*3
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
        
        if(output[0][0]-output[0][1]<-0.000001):
            output = 1
        elif(output[0][0]-output[0][1]>0.000001):
            output = 2
        else:
            output = 0  
        
        if(output==row['label']):
            correct+=1

        if(index>0 and index%500==0):
            print("Accuracy: ", float(correct/index))
    return float(correct/test_data.shape[0])


# training()

batch_size = 32
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

    accuracy = testing(test)
    print("Accuracy after epoch is :" , accuracy)
