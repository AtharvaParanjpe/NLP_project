from transformers import BertTokenizer, TFBertModel,TFBertForSequenceClassification
import numpy as np
import pandas as pd
import tensorflow as tf
import math


dev = pd.read_csv("subtask-2/modified-dev.csv")
train = pd.read_csv("subtask-2/modified-train.csv")
# test = pd.read_csv("subtask-1/modified-test.csv")
test = pd.read_csv("subtask-2/modified-test.csv")

class Subtask1_Model(tf.keras.Model):
    def __init__(self):
        super(Subtask1_Model, self).__init__()
        self.bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
        # self.dense_layer3 = tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32')
        self.dense_layer2= tf.keras.layers.Dense(256, activation="sigmoid",dtype='float32',
                                                kernel_regularizer=tf.keras.regularizers.l2(0.01))
#         self.dense_layer2= tf.keras.layers.Dense(128, activation="sigmoid",dtype='float32')
        self.dense_layer1 = tf.keras.layers.Dense(2, activation="sigmoid",dtype='float32')
#         self.dropout = tf.keras.layers.Dropout(0.4)

    def __call__(self, x_train1, x_train2):
        in1 = self.bert_layer(x_train1)
        in2 = self.bert_layer(x_train2)
#         intermediate2 = self.dropout(intermediate[1])
#         output1 = self.dense_layer2(intermediate2)
#         output2 = self.dense_layer1(output1)
        temp = tf.concat([in1[1], in2[1], tf.abs(tf.math.subtract(in1[1], in2[1])), tf.math.multiply(in1[1], in2[1])], 1)
        intermediate = self.dense_layer2(temp)
#         output1 = self.dropout(intermediate2)
        output2 = self.dense_layer1(intermediate)
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
    

def training(train_data):
    
    loss = 0
    input_ids1 = [] 
    input_ids2 = [] 
    attn_mask1 = []
    segment_ids1 = []
    attn_mask2 = []
    segment_ids2 = []
    targets = []
    labels = []
#     target1 = []
#     target2 = []

    for index, row in train_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids1.append(in_id1)
        attn_mask1.append(attn1)
        segment_ids1.append(seg1)
#         target1.append(row["meanGrade1"]/3)
        
        in_id1 = tokenizer.encode(row['original'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))
        input_ids2.append(in_id1)
        attn_mask2.append(attn1)
        segment_ids2.append(seg1)
#         target2.append(row["meanGrade2"]/3)
        
        targets.append([row["meanGrade1"]/3, row["meanGrade2"]/3])
#         labels.append(row["label"]/2)
        
    input_dictionary1 = {}
    input_dictionary2 = {}

    with tf.GradientTape() as tape:
        input_dictionary1['input_ids'] = tf.convert_to_tensor(np.array(input_ids1))
        input_dictionary1['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask1))
        input_dictionary1['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids1))
        
        input_dictionary2['input_ids'] = tf.convert_to_tensor(np.array(input_ids2))
        input_dictionary2['attention_mask'] = tf.convert_to_tensor(np.array(attn_mask2))
        input_dictionary2['token_type_ids'] = tf.convert_to_tensor(np.array(segment_ids2))
        
#         output = model(input_dictionary)
#         output1 = model(input_dictionary1)
#         output2 = model(input_dictionary2)
        output = model(input_dictionary1, input_dictionary2)
        targets = tf.convert_to_tensor(np.array(targets), dtype="float32")

#         target1 = tf.convert_to_tensor(np.array(target1), dtype="float32")
#         target1 = tf.reshape(target1 ,shape=(target1.shape[0],1))
        
#         target2 = tf.convert_to_tensor(np.array(target2), dtype="float32")
#         target2 = tf.reshape(target2 ,shape=(target2.shape[0],1))
        
        
#         labels = tf.convert_to_tensor(np.array(labels), dtype="float32")
#         labels = tf.reshape(labels ,shape=(labels.shape[0],1))

        # beforeThreshold = tf.math.subtract(targets,output)
        
#         result1 = tf.math.subtract(target1,output1)
#         result2 = tf.math.subtract(target2,output2)
        
#         squaredError1 = tf.math.square(result1)
#         squaredError2 = tf.math.square(result2)
        
#         sumSquaredError = tf.math.add(squaredError1, squaredError2)
        
#         loss = tf.math.reduce_mean(sumSquaredError)
        
#         loss = tf.math.reduce_mean(tf.math.reduce_sum(squaredError, axis=1))#,squaredError.shape[0])

        result = tf.math.subtract(targets,output)
        squaredError = tf.math.square(result)
        
        loss = tf.math.reduce_mean(tf.math.reduce_sum(squaredError, axis=1))
    
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
    input_dictionary1 = {}
    input_dictionary2 = {}
    correct = 0
    targets = []
    for index, row in test_data.iterrows():
        in_id1 = tokenizer.encode(row['original'], row["edit1"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        # targets = [row["meanGrade1"], row["meanGrade2"]]

        input_dictionary1['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary1['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary1['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
#         output1 = model(input_dictionary)
#         output1 = output1*3
        
        
        
        
        in_id1 = tokenizer.encode(row['original'], row["edit2"], add_special_tokens=True)
        attn1,seg1 = get_bert_params(tokenizer.convert_ids_to_tokens(in_id1))
        in_id1 += [0]*(maxLengthPadding-len(in_id1))

        # targets = [row["meanGrade1"], row["meanGrade2"]]

        input_dictionary2['input_ids'] = tf.convert_to_tensor(np.array(in_id1))[None,:]
        input_dictionary2['attention_mask'] = tf.convert_to_tensor(np.array(attn1))[None,:]
        input_dictionary2['token_type_ids'] = tf.convert_to_tensor(np.array(seg1))[None,:]
        
#         output2 = model(input_dictionary)
#         output2 = output2*3
        output = model(input_dictionary1, input_dictionary2)
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
        
        if(output[0][0]-output[0][1]<-0.0001):
            output = 2
        elif(output[0][0]-output[0][1]>0.0001):
            output = 1
        else:
            output = 0  
        
        if(output==row['label']):
            correct+=1

        if(index>0 and index%500==0):
            print("Accuracy: ", float(correct/index))
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
