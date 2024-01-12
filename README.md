## Applying machine learning on x-rayed, cleared, and fossil leaves for taxonomic classification of plants.
## Team: Sontosh Deb, Sarah Oladejo, Dang-Son Nguyen

### 1. Introduction:

The taxonomic identification of species is very difficult yet important for understanding earth’s plant diversity. Botanists often use fruits, flowers, seeds, and leaves for taxonomic classification of plants. Leaves are available more compared to the other plant organs as fresh specimens or even as fossils. However, leaves come in a wide variety of sizes and shapes and identification is very challenging even for a trained taxonomist. There are millions of preserved and fossil leaves collections available in the Museums of Natural History around the world. Fossil leaves are also often found in natural sites. These resources could provide important insights into plant evolution. However, they are underused in plant evolutionary studies. One of the main reasons behind this is their identification problems.
Recent applications of machine learning techniques on herbarium (a collection of preserved plant specimens) images for species identification showed promising results (e.g. Wilf et al., 2016; Carranza-Rojas et al., 2017; Seeland et al., 2019). Different plant traits from herbarium specimens were used to train several supervised learning and deep learning models. However, one of the main challenges of using herbarium samples is that the dataset produced is very prone to be biased because of the ways in which the training datasets are built. Besides herbarium specimens, a large resource database of cleared, x-rayed leaves, and fossil leaves are available. The existing datasets provide an unprecedented opportunity to develop novel machine learning algorithms of analyzing image datasets and to use existing techniques to derive information from these images for useful botanical implications. An open access database of 30,252 (26, 176 cleared + x-rayed leaves, and 4,076 fossil leaves) image dataset has been released very recently (Wilf et al., 2021). The dataset is particularly curated for human and machine learning. There is a potential to leverage the leaf architecture (i.e., leaf shape and venation) data from the image dataset and use them for training machine learning models for taxonomic classification. Computer vision followed by support vector machine were previously used on similar datasets (Wilf et al., 2016).
In our study, we leveraged images from the recently released Wilf et al. (2021) dataset to understand complex leaf patterns and utilize them for classifying plants at family level. We focused on identifying plants at families because many of the fossil species or even genera are now extinct and it is not possible to classify leaves at species level while including fossil leaves in the model training. However, there is potential to classify them at higher taxonomic groups for instance family or order level. We used different transfer learning and convolutional neural network approaches for our study and achieved promising results in plant’s taxonomic classification.The study has a great potential on taxonomic identification, understanding plant diversity, and evolution.

### 2. Methods

#### 2.1. Data Preprocessing

We retrieved the assembled image dataset by Wilf et al., (2021). Our dataset consisted part of the cleared and x-rayed leaves, fossil leaves. We filtered the dataset to contain only families with at least fifty representative images. We had a total of ten families and 3,158 images.


![Picture1](https://github.com/skdeb/MachineLearningProject/assets/53913657/70c3af60-c58c-49d0-a7d8-3decf5dd7e5f)


Figure 1: An example of an image dataset representing both cleared and fossil leaves to be used in the study. (Adapted from Wilf et al., 2021).

We used a python module “split-folders” to split the dataset into 80%, 10%, and 10% as train, validation, and test sets respectively. ImageDataGenerator from Keras API was used for rotation and horizontal flip of the training and validation set as part of data augmentation. We also imported “preprocess input” specific for Inception-ResNet-v2 along with the model and used for preprocessing such as resize and rescaling of the train, validation, and the test images. We used batch size of 32, target input size 299, 299 as image width and height, RGB color mode, and categorical class mode while loading the dataset for training, validating, and testing the model. We also shuffled the images using a fixed seed value.

#### 2.2. Deep learning API, platform, and network architecture

To achieve the aim of this project, we applied convolutional neural network based approaches on image dataset using Tensorflow library with Keras API as interface. The use of CNNs is an offshoot of knowledge of the brain’s visual cortex and has been applied for image recognition since the 1980s. CNNs use partially connected layers and weight sharing and are therefore good for large image datasets. CNN classifies images by capturing only pixels in the receptive field of input images in the first convolutional layer. Only a small rectangle in the first layer determined by filter size is passed to the next layer and so on.
For our study we used Inception-ResNet-v2 architecture (Figure 2) (Szegedy et al., 2016). The choice of this architecture is based on the consideration that the inception module is computationally efficient and the ResNet module has residual connections that provide optimization benefits (Seeland et al., 2019). In addition, other plant leaf image classification studies achieved high accuracy using this network architecture. We used the ‘imagenet’ weights and kept the layer of the base model
non-trainable. We added a few layers with the base model output which include a global average pooling layer, a dense layer with ‘ReLu’ activation function, and a dropout layer. Finally, the output layer was provided with a ‘softmax’ activation function to predict our studied ten plant families. We compile the model using ‘adam’ as optimizer, ‘categorical cross entropy’ as loss function, and accuracy as performance metrics. The model was trained for 30 epochs with steps per epoch and validation steps equal to length of the training set and validation set respectively.


Figure 2: Inception-Resnet-v2 network architecture (Adapted from Bhatia et al., 2019).
 
#### 2.3. Model Benchmarking and Comparison

To compare our model with other established models for leaf classification, we tried some other CNN architectures including a deep convolutional neural network, transfer learning with ResNet50, AlexNet, and LeNet (used by Liu et al. 2018 for leaf classification). Batching and prefetching was added to all the datasets.

##### 2.3.1. Deep CNN:

Here we stacked a few convolutional layers followed by Max pooling layers and dense layers at the end. We used a kernel size of 3, ReLu activation function, and an output layer with the softmax activation function. The model was compiled using sparse categorical cross entropy as loss, adam as optimizer, metrics as accuracy and trained for thirty epochs.

##### 2.3.2. AlexNet :

The dataset used for AlexNet was restructured to 227 x 227 without initial augmentation process as this was found to affect performance of the model. The AlexNet architecture had five convolutional layers with varying kernel sizes and strides with three Maxpooling layers and 3 fully connected dense layers towards the end of the architecture. Overall, the architecture is eight layers deep, the max pooling layers help to downsample the weight and height of the tensors and keep the depth the same. The architecture uses the ReLu activation function. We compiled the model using adam with a learning rate of 0.001, sparse categorical cross entropy as loss and accuracy as performance metrics. The model was also trained for 30 epochs.

##### 2.3.3. LeNet:

LeNet was developed by Yann Le-Cun et al. in the late 1990s for classifying grayscale images. LeNet-5 consists of seven layers including two convolutional layers of different filter sizes and kernel size of 5 x 5. Each convolutional layer was followed by a max pooling layer. The last 3 layers were fully connected dense layers. Our LeNet architecture used the ReLu activation function and a softmax activation function for the output layer. We trained for 30 epochs and used a learning rate scheduler (ReduceLRonPlateau) to reduce learning rate when validation error stops dropping. We compiled the model using the adam optimizer, sparse categorical cross entropy as loss and accuracy as performance metrics. The model was also trained for 30 epochs.

##### 2.3.4. ResNet 50:

Finally, we tried the ResNet 50 pre-trained model on our dataset. The ResNet architecture adds skip connections to the model to improve performance. It consists of a
stack of residual units which are each small neural networks with batch normalization, ReLu activation, kernel size of 3 x 3 with skip connections to alleviate the problem of vanishing gradients. ResNet-50 is a variant of ResNet with 50 layers. We loaded the pretrained network using Keras and specified image nets as weights to be used. We added a GlobalAverage pooling layer and dense layer to the base model. We resized our training and validation set to 224 x 224 which is expected by the ResNet 50 algorithm. First, we froze the initial layer to help the top layers learn some useful weight and trained for 30 epochs. Next, we set trainable layers to True and resumed training. The model was compiled using adam as optimizer, sparse categorical cross entropy as loss and accuracy as performance metrics. The model was also trained for 30 epochs.

### 3. Results and Discussion

Our approach achieved 71% training accuracy and 72% validation accuracy (Figure 3a) using the Inception-Resnet-v2 architecture. However, the accuracy using the test dataset was 68%. The training and validation loss were both around 0.79 (Figure 3b). Wilf et al. (2016) study using only cleared leaves found 72% accuracy for classifying plants at family and order levels. They used around 7500 images representing two groups where one is 100 images in each of 19 families and the other is 50 images in each of 29 families. They also classify at 19 order levels (up one in the order of taxonomic group than family) with 50 images in each order. The inclusion of fossil leaves (contains more noise in the images) and lower number of images in our study could be a potential reason to have lower test accuracy. Other studies using different types of leaf image dataset achieved considerably higher accuracy in the identification of species (e.g. Carranza-Rojas et al., 2017; Seeland et al., 2019). Training the models with higher number of epochs and fine tuning of the transfer model was found to provide better results. The use of more computational resources in our study has the potential to increase the model performance.

 
Figure 3: a) Training and validation accuracy in each epoch. b) Training and validation loss in each epoch. In both cases validation accuracy and loss fluctuated more compared to training sets.

Among other models tested in our dataset, ResNet 50 performed well in both cases of training and validation sets (Table 1). However, the test dataset accuracy was 68%. In addition the loss was much higher in ResNet 50 compared to
Inception-ResNet-v2. All other models showed very high accuracy in the training set while the validation accuracy was very low (Table 1). The use of a very small dataset could be a potential reason for model overfitting. However, hyperparameter tuning and use of more leaf images has the potential to get more robust prediction from these tested network architectures.



### 4. Conclusion

We trained a model that will help to classify the plant images into their specific family with considerable accuracy. The use of herbarium specimens for classifying plant species is not completely new and several models have provided promising results in the past. However, the use of cleared, x-rayed, and fossil leaves for species identification is lower compared to herbarium specimens. The leaf images used in our study contain more noise than fresh or herbarium specimens. In addition, classifying plants at higher taxonomic groups such as families is also more challenging compared to species identification. Using a large number of images from the database could provide better results. The availability of more computational resources would also allow better hyperparameter tuning and potentially result in better model performance. In our future studies, we aim to increase the number of leaf images from the database and also the number of plant families. In addition, we will apply image segmentation to remove uninformative background or convert images to binary format, and use other network architecture to achieve better prediction accuracy of plant families. The development of a model with high prediction accuracy of cleared, x-rayed, and fossil leaves has great implications in evolutionary botany.

### References

Bhatia, Y., Bajpayee, A., Raghuvanshi, D. and Mittal, H., 2019, August. Image Captioning using Google's Inception-resnet-v2 and Recurrent Neural Network. In 2019 Twelfth International Conference on Contemporary Computing (IC3) (pp. 1-6). IEEE.

Carranza-Rojas, J., Goeau, H., Bonnet, P., Mata-Montero, E. and Joly, A., 2017. Going deeper in the automated identification of Herbarium specimens. BMC evolutionary biology, 17(1), pp.1-14.

Liu, J., Yang, S., Cheng, Y. and Song, Z., 2018, November. Plant leaf classification based on deep learning. In 2018 Chinese Automation Congress (CAC) (pp. 3165-3169). IEEE.

Seeland, M., Rzanny, M., Boho, D., Wäldchen, J. and Mäder, P., 2019. Image-based classification of plant genus and family for trained and untrained plant species. BMC bioinformatics, 20(1), pp.1-13.

Szegedy C., Ioffe S., Vanhoucke V., and Alemi AA., 2016. Inception-v4, inception-resnet and the impact of residual connections on learning. In: ICLR 2016 Workshop.

Wilf, P., Wing, S.L., Meyer, H.W., Rose, J.A., Saha, R., Serre, T., Cúneo, N.R., Donovan, M.P., Erwin, D.M., Gandolfo, M.A. and González-Akre, E., 2021. An image dataset of cleared,
x-rayed, and fossil leaves vetted to plant family for human and machine learning. PhytoKeys, 187, p.93. https://doi.org/10.25452/figshare.plus.14980698

Wilf, P., Zhang, S., Chikkerur, S., Little, S.A., Wing, S.L. and Serre, T., 2016. Computer vision cracks the leaf code. Proceedings of the National Academy of Sciences, 113(12), pp.3305-3310.
