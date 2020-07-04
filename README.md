# Image-Caption-Generator-using-ResNet50-and-LSTM-model
## 1.0 Model Development ## 
### 1.1 Model Architecture
A merge-model architecture is used in this project to create an image caption generator. In this
model, the encoded features of an image are used along with the encoded text data to generate the
next word in the caption. In this approach, RNN is used only to encode text data and is not
dependent on the features of the image. After the captions have been encoded, those features are
then merged with the image vector in another multimodal layer which comes after the RNN
encoding layer. This architecture model provides the advantage of feeding preprocessed text data
to the model instead of raw data. The model has three major blocks:
<ul>
<li>Image feature extractor</li>
<li>Text processor</li>
<li>Output predictor</li>
</ul>

### 1.2 Image Feature Extractor<br>
The feature extractor needs an image 224x224x3 size. The model uses ResNet50 pretrained on
ImageNet dataset where the features of the image are extracted just before the last layer of
classification. Another dense layer is added and converted to get a vector of length 2048. 


### 1.3 Text Processor<br>
This layer contains the word embedding layer to encode text data. Long Short-Term Memory
(LSTM) in RNN is added with 256 memory units. This unit also outputs a vector of length 128.

### 1.4 Output Predictor
Output vector from both the image feature extractor and the text processor are of same length (128)
and a decoder merges both the vectors using an addition operation. This is then fed into two dense
layers. The first layer is of length 128 and the second layer makes a prediction of the most probable
next word in the caption. This layer uses softmax activation function to predict the most probable
next word in the vocabulary.

### 1.5 Fitting the Model
After building the model, the model is fit using the training dataset. The model is made to run for
210 epochs and the best model is chosen among the 210 epochs by computing loss function on
Flickr8k development dataset. The model with the lowest loss function is chosen for generating
captions.

## 2.0 Working ##
It involves various phases including Importing the dataset and preprocess, Image Resizing and
feature extraction, Vocabulary definition and indexing, Data generator, Training the model,
Testing the model and Caption Generation

### 2.1 Importing the dataset and preprocess
All the data that we will work upon has been imported and we have performed the basic cleaning
and preprocessing. The textual data has been sorted and each image number and its corresponding
caption has been segmented intp individual rows. A dictionary is made to store the images along
with their captions, where the images are keys and the captions serve as values. All the 5 captions
are stored for each image. Also, the dataset has been divided into 3 –train, validation and test sets
and 3 files are made respectively containing the images and captions for each set. Next, we have
added, <start> and <end> sequence tokens before and after each caption and stored in encoded
version in the files.

### 2.2 Image Resizing and feature extraction
For feature extraction, we have used a pretrained model, i.e. we have used transfer learning
technique. We have used ResNet50 model for the same with weights as Imagenet having already
pre-trained models on standard Imagenet dataset provided in keras. It is a standard dataset used for
classification and contains more than 14 million images in the dataset, with little more than 21
thousand groups or classes. However, our aim is not to classify, but just get fixed-length
informative vector for each image. This process is called automatic feature engineering. Then
this vector will be used for the generation of the caption.
But the images in the dataset are of different shapes and sizes whereas the feature extraction model
takes only a fixed size vector. To input images into the feature extraction model, we have to resize
them into target size which in our case is ResNet50 which takes 224x224x3 as input size.
From our model, we have removed soft-max layer to use it as a feature extractor. For a given input
image, ResNet50 gives us 2048-dimensional feature extracted vector.
All the extracted features are stored in a Python dictionary and saved on the disk using Pickle file,
namely whose keys are image names and values are corresponding 2048 length feature vector.

### 2.3 Vocabulary definition and indexing
To define the vocabulary from which the captions for the input images will be predicted, we had
to find the words from all the captions i.e. we need to tokenize each caption and list all unique
words from all them. From the training dataset we made a vocabulary of 8253 words. As computers
do not understand English words, we have represented them with numbers and mapped each word
of the vocabulary with a unique index value. Also to decide the language model parameters, we
need to specify each caption in a fixed size, hence we calculated the maximum length of the
captions. Max_length of caption is 40.

### 2.4 Data generator
The prediction of the entire caption, given the image does not happen at once. It is done word by
word. Thus, we encoded each word into a fixed sized vector and represented each word as a
number. The main purpose of this module is to generate various formats for images and captions,
which could be further used in various models while implementing the training of data. Firstly, is
capsule is considered and it is splitted into words and it referenced to its corresponding index.
Then, beginning from the '<start>' prefix, each time we append next word in the partial sequence,
the number of elements in the same are incremented by 1. Also, we maintain a list for each caption
that stores the next word at each sub-iteration. Further, one hot encoding is applied on the list that
contains the next word. Further, both partial sequence and one hot encoded next word are
converted into arrays.

Then, keeping in mind space and time complexities, only '2000’ images and their captions are
considered. The formats of this images and captions are replicated and manipulated, which leads
to generation of new formats of images and captions. Further, different files are created and,
images and captions in different forms are stored respectively in these files.

### 2.5 Training the model
In the training of the model, we first apply the Sequential model which contains a Dense layer that
uses 'relu' as the element-wise activation function. Then we add a Repeat vector layer with
argument '40’, which would repeat the input 40 times.
Then we apply another sequential model, in which we use the Embedding layer as the first layer
of the model. In this layer, the input dimension is the size of the vocabulary and the output
dimension, i.e., the dimensions of the dense embedding is declared to be '128’. Further we will add
a LSTM layer with '256’ as the dimensions of the output space and the Boolean ‘return_sequences’
is set TRUE, which would return the full sequence and not just the last output. Then there is a
TimeDistributed layer whose argument is a regular deeply connected neural network of size '128’.
This layer will help in applying a layer to every temporal size of an input.
After defining the above two sequential models, we concatenate them, followed by addition of two
LSTM layers with output dimensionality equal to '128’ and '512’ respectively. Then we will apply
a Dense layer, that would deeply connect the network with size equal to the vocabulary size. Then
the fully connected layer reduces its input to number of classes using softmax activation
For the we train the model by passing the images as a list whose dimensions were reshaped after
applying the ResNet50 model; and captions as a list of list which would store the the sequence in
which a caption is generated word by word for each image. Here, the batch size, i.e., the number
of samples trained in a single iteration is defined as '512’ and the number of epochs, i.e., the number
of times the entire dataset is pass both forward and backward is set to ‘210’. 
At last, the result of the training of the model is important in the form of weights and is also saved
to a file for future reference.

### 2.6 Testing the model and Caption Generation
To test our trained model, we input an image to the model. Next the image is fed into the feature
extractor to recognize what all objects and scenes are depicted in the image, after resizing it. The
process of caption generation is done using the RNN trained model. Then for that image,
sequentially, word-by-word the caption is generated by selecting the word with maximum weight
for the image at that particular iteration. The indexed word is converted to word and then appended
into final caption. When <end> tag is detected or the size of the caption reaches 40, the final
caption is generated and printed along with the input image.
