# // this was coded in google colab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import ReLU,LeakyReLU,ELU,PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from transformers import BertTokenizer,TFBertModel
from tensorflow.keras.layers import LSTM,Bidirectional

df=pd.read_csv('dataset_train.csv')

df

"""##BERT Approach"""

#Lets Use Bert
tokenizer= BertTokenizer.from_pretrained('bert-base-uncased')
bert_model=TFBertModel.from_pretrained('bert-base-uncased')

questions=df['Question'].tolist()
inputs=tokenizer(questions,return_tensors='tf',padding=True,truncation=True,max_length=512)

embeddings=bert_model(inputs['input_ids'])['last_hidden_state']

#for classification, take mean embeddings....
X=tf.reduce_mean(embeddings,axis=1)
X.shape

"""## Encoding the Y !"""

y_raw=df['Type']
# y_raw

"""Label Encoding"""

le=LabelEncoder()
y_le=le.fit_transform(y_raw)
# y_le

"""One Hot Encoding"""

ohe=OneHotEncoder()
y_ohe=ohe.fit_transform(y_raw.values.reshape(-1,1))
# y_test2=ohe.transform(y_test2[0].values.reshape(-1,1))
# print(y_ohe.toarray())

# print(y_test2.shape)

"""Selection Between Label and One Hot encoding
----------------------
Also accordingly need to make changes in the Output Layer
"""

#One Hot
y=pd.DataFrame(y_ohe.toarray())
#-------
#Label
# y=pd.DataFrame(y_le)

#If Label Encoding is used
# y=y.rename(columns={0:'y1',1:'y2',2:'y3',3:'y4',4:'y5',5:'y6'})

"""## Train Test Split"""

X=X.numpy()

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.33)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

time_steps=1
samples = X_train.shape[0]
features=X_train.shape[1]

X_train=X_train.reshape(samples,time_steps,features)
X_test=X_test.reshape(X_test.shape[0],time_steps,X_test.shape[1])

"""## THE ANN Approach"""

#Early Stopping
stopper=EarlyStopping(
    monitor='val_loss',
    min_delta=0.1,
    patience=5,
    verbose=2,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
)

# # model=Sequential()

# ## Input Layer , 20 features.
# model.add(Dense(units=20,activation='relu'))

# # 1st Hidden Layer
# model.add(Dense(units=28,activation='relu'))
# model.add(Dropout(0.3))

# #2nd Hidden Layer
# model.add(Dense(units=32,activation='relu'))
# model.add(Dropout(0.24))

# #3rd hidden layer
# model.add(Dense(units=26,activation='relu'))
# model.add(Dropout(0.35))

# #Output Layer
# model.add(Dense(units=5,activation='softmax'))

learning=0.01
opt=Adam(learning_rate=learning)

# #Optimizer and Loss function
# model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
# model_history=model.fit(X_train,y_train,batch_size=5,epochs=100,validation_split=0.33,callbacks=stopper)

# model_history.history.keys()

# plt.plot(model_history.history['accuracy'])
# plt.plot(model_history.history['val_accuracy'])
# plt.xlabel('Epochs')
# plt.legend(['Train','Validation'])
# plt.show()
# plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_loss'])
# plt.xlabel('Epochs')
# plt.legend(['Train','Validation'])
# plt.show()

# y_pred=model.predict(X_test)
# y_pred_op=np.argmax(y_pred,axis=1)
# y_test_op=np.argmax(y_test,axis=1)

# acc_score=accuracy_score(y_pred_op,y_test_op)

# cm=confusion_matrix(y_pred_op,y_test_op)

# cr=classification_report(y_pred_op,y_test_op)
# print(cr)

# # Reshape X_train and X_test for LSTM
# samples, features = X_train.shape

# X_train = X_train.reshape(samples, 1, features)
# X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])




# opt = Adam(learning_rate=0.001)

"""## Bidirectional LSTM approach"""

model=Sequential()

#adding the lstm layer
model.add(Bidirectional(LSTM(units=768,return_sequences=False,dropout=0.3,recurrent_dropout=0.3)))

model.add(Dense(units=384,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(units=192,activation='relu'))
model.add(Dropout(0.2))

# model.add(Dense(units=96,activation='relu'))
# model.add(Dropout(0.35))

model.add(Dense(units=48,activation='relu'))
model.add(Dropout(0.2))

# model.add(Dense(units=24,activation='relu'))
# model.add(Dropout(0.5))

# model.add(Dense(units=12,activation='relu'))
# model.add(Dropout(0.4))

model.add(Dense(units=5,activation='softmax'))
model.add(Dropout(0.2))

# model.add(Dense(units=3,activation='relu'))
# model.add(Dropout(0.4))

# #adding the output layer
# model.add(Dense(units=1,activation='softmax'))

# optimizer and loss function
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#Lets Train the Model
model_history=model.fit(X_train,y_train,callbacks=stopper,epochs=120)

# print(df.isnull().sum())

# model_history.history.keys()

''' Analyzing the results 


plt.plot(model_history.history['accuracy'])
# plt.plot(model_history.history['val_accuracy'])
plt.title("accuracy")
plt.xlabel("epoch")
plt.legend(['train','test'],loc='upper left')
# plt.show()

## Summary history for accuracy
plt.plot(model_history.history['loss'])
# plt.plot(model_history.history['val_accuracy'])
plt.title("loss")
plt.xlabel("epoch")
plt.legend(['loss','test'],loc='upper left')
# plt.show()


'''
y_pred=model.predict(X_test)

y_pred=np.argmax(y_pred,axis=1)
# y_test=np.argmax(y_test,axis=1)

# print(y_pred)

input_op=["How can i apply for scolarship in sars"]
question_op=input_op
inputs_op=tokenizer(question_op,return_tensors='tf',padding=True,truncation=True,max_length=512)
embedding_op=bert_model(inputs_op['input_ids'])['last_hidden_state']

X_op=tf.reduce_mean(embedding_op,axis=1)
X_op.shape
X_op=X_op.numpy()

time_steps=1
samples = X_op.shape[0]
features=X_op.shape[1]

X_op=X_op.reshape(X_op.shape[0],time_steps,X_op.shape[1])
# X_op.shape

y_op=model.predict(X_op)

y_op=np.argmax(y_op,axis=1)

# y_op[0]

# ohe.categories_[0]

print(ohe.categories_[0][y_op])

