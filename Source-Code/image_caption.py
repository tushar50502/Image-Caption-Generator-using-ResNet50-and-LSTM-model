# importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import numpy as np
import os
import glob
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
import random
from keras.preprocessing import image, sequence
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')
import glob

images_directory = '/content/drive/My Drive/Flickr_Data/'

img_path = '/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/'
cap_path = '/content/drive/My Drive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
training_path = '/content/drive/My Drive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
valid_path = '/content/drive/My Drive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
testing_path  = '/content/drive/My Drive/Flickr_Data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

cap = open(cap_path, 'r').read().split("\n")
x_training = open(training_path, 'r').read().split("\n")
x_valid = open(valid_path, 'r').read().split("\n")
x_testing = open(testing_path , 'r').read().split("\n")

# Loading cap as values and images as key in dictionary
tok = {}

for item in range(len(cap)-1):
    tem = cap[item].split("#") #tem[0]= imgname.jpg ..... tem[1]=0  captionn.
    if tem[0] in tok:
        tok[tem[0]].append(tem[1][2:])
    else:
        tok[tem[0]] = [tem[1][2:]] #tem[n]= imgName ... #tok[tem[n]] = list of caption

# Making 3 files with 2 colmns as 'image_id' and 'captions'
training_dataset = open('flickr_8k_train_dataset.txt','wb')
training_dataset.write(b"image_id\tcap\n")

valid_dataset = open('flickr_8k_val_dataset.txt','wb')
valid_dataset.write(b"image_id\tcap\n")

testing_dataset = open('flickr_8k_test_dataset.txt','wb')
testing_dataset.write(b"image_id\tcap\n")

# Loading image ids and captions for each of these images in the above 3 files
for img in x_training:
    if img == '':
        continue
    for capt in tok[img]:
        caption = "<start> "+ capt + " <end>"
        training_dataset.write((img+"\t"+caption+"\n").encode())
        training_dataset.flush()
training_dataset.close()

for img in x_testing:
    if img == '':
        continue
    for capt in tok[img]:
        caption = "<start> "+ capt + " <end>"
        testing_dataset.write((img+"\t"+caption+"\n").encode())
        testing_dataset.flush()
testing_dataset.close()

for img in x_valid:
    if img == '':
        continue
    for capt in tok[img]:
        caption = "<start> "+ capt + " <end>"
        valid_dataset.write((img+"\t"+caption+"\n").encode())
        valid_dataset.flush()
valid_dataset.close()

# Here, we're using ResNet50 Model
from IPython.core.display import display, HTML
model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
model.summary()

#  process images to target size
def preprocess(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im) # (x, y, z)
    im = np.expand_dims(im, axis=0)  # (0, x, y, z)
    return im

training_data = {}
counter=0
for item in x_training:
    if item == "":
        continue
    if counter >= 3000:
        break
    counter+=1
    if counter%1000==0:
        print(counter)
    path = img_path + item
    img = preprocess(path) #to change the dimensions of the image for using ResNet model
    pred = model.predict(img).reshape(2048)  # shape of each image is (2048, 0)
    training_data[item] = pred

# opening train_enc_img.p file and dumping content of training_data to this file
with open( "train_enc_img.p", "wb" ) as pickle_f: #obj hierarchy is converted into byte stream
    pickle.dump(training_data, pickle_f )

# Storing image and its corresponding caption into a dataframe 
pd_dataset = pd.read_csv("flickr_8k_train_dataset.txt", delimiter='\t')
dframe = pd_dataset.values
print(dframe.shape)

pd_dataset.head()

# Storing all the captions from dframe into a list
senten = []
for item in range(dframe.shape[0]):
    senten.append(dframe[item, 1])
#senten will have 30000 length
# First 5 captions stored in senten
senten[:5]

# Splitting each captions stored in 'senten' and storing them in 'wor' as list of list
wor = [i.split() for i in senten]

# Creating a list of all unique wor
uniq = []
for i in wor:
    uniq.extend(i)
uniq = list(set(uniq))

print(len(uniq))

vocabulary_size = len(uniq)

# making 2 lists to index each unique word and vice-versa
w_to_i = {val:index for index, val in enumerate(uniq)}
i_to_w = {index:val for index, val in enumerate(uniq)}

w_to_i['UNK'] = 0
w_to_i['raining'] = 8253
i_to_w[0] = 'UNK'
i_to_w[8253] = 'raining'

vocabulary_size = len(w_to_i.keys())
print(vocabulary_size)

max_len = 0

for i in senten:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)

print(max_len) #finding longest caption

pad_seq, subsequent_wor = [], []

for item in range(dframe.shape[0]):  #30000 items
    part_seq = []
    next_wor = []
    text = dframe[item, 1].split() #diving each caption for every image into words
    text = [w_to_i[i] for i in text] #finding index for each word
    for i in range(1, len(text)):
        part_seq.append(text[:i])  #start, 1st word, ... , last word
        next_wor.append(text[i])   #1st word, ... , last word, end
    pad_part_seq  = sequence.pad_sequences(part_seq, max_len, padding='post')

    next_wor_1hot = np.zeros([len(next_wor), vocabulary_size], dtype=np.bool)
    
    for i,next_word in enumerate(next_wor):
        next_wor_1hot[i, next_word] = 1
        
    pad_seq.append(pad_part_seq )
    subsequent_wor.append(next_wor_1hot)
    
pad_seq = np.asarray(pad_seq)
subsequent_wor = np.asarray(subsequent_wor)

print(pad_seq.shape)
print(subsequent_wor.shape)

print(pad_seq[0])

for item in range(len(pad_seq[0])):
    for y in range(max_len):
        print(i_to_w[pad_seq[0][item][y]],)
    print("\n")

print(len(pad_seq[0]))

num_imgs = 2000

cap = np.zeros([0, max_len])
next_wor = np.zeros([0, vocabulary_size])

for item in range(num_imgs): #img_to_padded_seqs.shape[0]):
    cap = np.concatenate([cap, pad_seq[item]])
    next_wor = np.concatenate([next_wor, subsequent_wor[item]])

np.save("cap.npy", cap)
np.save("next_wor.npy", next_wor)

print(cap.shape)
print(next_wor.shape)

with open('train_enc_img.p', 'rb') as f:
    enc_img = pickle.load(f, encoding="bytes")

imgs = []

for item in range(dframe.shape[0]): #30000

    if dframe[item, 0] in enc_img.keys(): #dframe[0,0], [1,0], ... , [4,0] match with 0th key of enc_img
      
        imgs.append(list(enc_img[dframe[item, 0]])) 

imgs = np.asarray(imgs)
print(imgs.shape)

images = []
img_names = []

for item in range(num_imgs):  #2000
    for y in range(pad_seq[item].shape[0]): #14
        images.append(imgs[item]) #1st iteration: 14 times name of image in byte form
        img_names.append(dframe[item, 0]) # normal form
        
images = np.asarray(images) #images contains image_name in byte form
np.save("images.npy", images)

img_names = np.asarray(img_names) #img_names contains image_name normally
np.save("img_names.npy", img_names)

print(images.shape)
print(len(img_names))

cap = np.load("cap.npy")
next_wor = np.load("next_wor.npy")

print(cap.shape)
print(next_wor.shape)

images = np.load("images.npy")
print(images.shape)

imag = np.load("img_names.npy")
print(imag.shape)

embed_size = 128
max_len = 40

img_model = Sequential()

img_model.add(Dense(embed_size, input_shape=(2048,), activation='relu'))
img_model.add(RepeatVector(max_len))

img_model.summary()

lang_model = Sequential()

lang_model.add(Embedding(input_dim=vocabulary_size, output_dim=embed_size, input_length=max_len))
lang_model.add(LSTM(256, return_sequences=True))
lang_model.add(TimeDistributed(Dense(embed_size)))

lang_model.summary()

concat = Concatenate()([img_model.output, lang_model.output])
x = LSTM(128, return_sequences=True)(concat)
x = LSTM(512, return_sequences=False)(x)
x = Dense(vocabulary_size)(x)
out = Activation('softmax')(x)
model = Model(inputs=[img_model.input, lang_model.input], outputs = out)

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()

hist = model.fit([images, cap], next_wor, batch_size=512, epochs=210)

for label in ["loss"]:
    plt.plot(hist.history[label],label=label)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

for label in ["accuracy"]:
    plt.plot(hist.history[label],label=label)
plt.legend()
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()

model.save_weights("model_weights.h5")

def preprocess(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im) #(224,224,3)
    im = np.expand_dims(im, axis=0) #(1,224,224,3)
    return im

def get_encode(model, img):
    image = preprocess(img)
    pred = model.predict(image).reshape(2048)
    return pred

resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')

img = "/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/3376942201_2c45d99237.jpg"
test_img = get_encode(resnet, img)

def predict_cap(image):
    start_wor = ["<start>"]
    
    while True:
        par_cap = [w_to_i[i] for i in start_wor] #par_cap list is made
        par_cap = sequence.pad_sequences([par_cap], maxlen=max_len, padding='post') #convert list to sequence of len = 40
        preds = model.predict([np.array([image]), np.array(par_cap)]) # PREDICTION
        xx = np.argmax(preds[0])       
        word_pred = i_to_w[xx] # convert 5972 to DOG
        start_wor.append(word_pred) # [dog] is added in list
        
        if word_pred == "<end>" or len(start_wor) > max_len:
            break
            
    return ' '.join(start_wor[1:-1])

final_caption = predict_cap(test_img)

from IPython.display import Image,display
z = Image(filename=img)
display(z)

print(final_caption)

img = "/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/1.jpg"
test_img = get_encode(resnet, img)

from IPython.display import Image,display
final_caption = predict_cap(test_img)
z = Image(filename=img)
display(z)

print(final_caption)

img = "/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/car.jpg"
test_img = get_encode(resnet, img)

from IPython.display import Image,display
final_caption = predict_cap(test_img)
z = Image(filename=img)
display(z)

print(final_caption)

img = "/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/bike.jpg"
test_img = get_encode(resnet, img)

from IPython.display import Image,display
final_caption = predict_cap(test_img)
z = Image(filename=img)
display(z)

print(final_caption)

img = "/content/drive/My Drive/Flickr_Data/Flickr_Data/Images/tennis.jpg"
test_img = get_encode(resnet, img)

from IPython.display import Image,display
final_caption = predict_cap(test_img)
z = Image(filename=img)
display(z)

print(final_caption)
