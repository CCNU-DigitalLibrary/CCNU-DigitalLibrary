# CCNU-DigitalLibrary

## Research significance

This topic focuses on the semantic association method between query image and parsed text and query image search matching method in mobile visual search system with digital library as the research object. This topic intends to carry out research on the research object from the following four aspects, as follows:

### **Image data cleaning in digital libraries**

The generalization adaptability of the deep neural network is highly dependent on the number of image samples and the quality of annotation for training the deep neural network. There are a large number of unannotated media data in digital libraries, for example, the first phase of the CADAL project has built 1.023 million digital resources in English and Chinese, including digital media resources from ancient to modern documents. The quality of image data in digital libraries can be cleaned and annotated according to the national standard GB/T31219, which directly affects the effect of deep learning technology and the credibility of the conclusions drawn. The research is to clean and annotate the image data in the digital library, and to generate a sample image database for the research of semantic association method between query image and parsed text and search and matching method of query image.

### **Semantic association method between query image and parsed text**

The similarity of heterogeneous data cannot be directly measured by the underlying feature data because the underlying features of different media data are different (e.g., images are represented by pixels and texts are represented by words). How to mine the intrinsic connection of heterogeneous data and learn a common isomorphic subspace for heterogeneous data becomes a key problem for the research. The research approach should not only support the spanning of data types in the retrieval process, but also support the spanning of heterogeneous media data in semantic terms. For example, images of tigers and descriptive textual data of tigers are expressed in different forms, but the concept of tiger is expressed at the semantic level. Specifically A deep neural network model that can handle the semantic association between the query image and the parsed text is constructed. Second, a deep neural network model is constructed using A migration learning method for training the constructed deep neural network models with a small amount of annotated image data from digital libraries. Finally, the studied deep neural network model is used to generate an image-text association database as a database for query image search results. The database is used as a database for query image search results.

### **The query image search matching method**

Deep learning techniques are particularly suitable for processing big data from a statistical and computational point of view, as they are almost the only end-to-end machine learning systems that work directly on the raw data, automatically learning media features layer by layer, and optimizing an objective function directly throughout the process. Deep learning techniques, like probabilistic models, provide a rich, connectionist-based modeling framework that can express the rich relationships and structures inherent in the data, and are typically trained on elaborate multilayer neural networks for learning the underlying media features, leading to more discriminative high-level semantic descriptions. The hash indexing approach is to use hash functions to map the high-dimensional representation of image data to the Hamming space of the low-dimensional representation of data, while quantifying the data with binary numbers, which take up little storage space and can be obtained by a simple heterogeneous or operation during online retrieval of query images.Similarity. In this study, a deep neural network model is constructed to generate a hash coding of semantic features directly from the query image. Secondly, we use a small amount of annotated image data from digital libraries to train the constructed deep neural network The migration learning method of the model.

### **Development and empirical validation of method validation software**

At present, a large number of deep learning tools are developed and shared in the form of open source and free of charge, and the representative open source tools for deep neural networks are Caffe, TensorFlow, etc. For example, Nvidia also provides a free GPU-based CUDA framework for scientific computing and a cuDNN library designed for neural networks to facilitate the application of deep learning technology, which greatly reduces the threshold for general users to use deep learning technology. The specific research is to develop a validation software adapted to the research results of this project using existing open source tools. Secondly, the open image and text data in the digital library are used to compare the search results in terms of completeness and accuracy to verify the effectiveness and practicality of the mobile visual search method studied in this project.



## Research Content

The project is divided into four parts:


## Cross-modal Image-text Retrieval Based on Multi-modal Mask and Hash Constraint


The model proposed in this project is based on Transformer architecture and the results obtained by using multi-modal mask and hash constraint techniques show that the cross-modal retrieval performance is improved, which is the leading algorithm of its kind and proves the effectiveness of the designed structure.


## An Image Retrieval Model with Attention-Based Deep Hashing


Abstract—Deep hashing greatly reduced the calculation amount of nearest neighbor search, avoids dimension disaster, and reduces the cost of data storage. However, the current deep hashing methods directly extract the global features of the image, resulting in a waste of resources. This thesis studies the existing deep hashing methods, adopts the idea of pairwise similarity measure learning, and improves the end-to-end hash generation network to further enhance the effect of image retrieval. The main work of this thesis is to improve the structure of the existing deep hashing network, embed the attention module into the feature extraction process, and construct the Attention Deep Hashing Network (ADHN). The Squeeze-and-Excitation module is improved by using the Gaussian error linear unit activation function, and the SE-G attention module is proposed. And through experiments with different positions and different parameter combinations, it is determined that the SE-G attention module is added after the first and third stages of ResNet50 to implement weight application and reduce feature dimensions. ADHN focuses more intensively on the important information needed for image retrieval during feature extraction, thus further improving the accuracy and efficiency of image retrieval. The experimental results show that on the three public data sets of CIFAR-10, VOC2012 and Flickr25K, ADHN has achieved higher mAP, precision and recall in 16bits, 32bits, 48bits and 64bits hash image retrieval. In terms of mean training time and number of network parameters, ADHN is not inferior to other deep hashing models due to the introduction of SE-G attention module. 


## Image-Text Cross-modal Retrieval Baesd on Improved Generative Adversarial Network	


For the initial model SCH-GAN, the similarity between the multimodal data is not well considered, and the dissimilar data is not well separated. In order to solve this problem, a semi-supervised cross-modal hash retrieval based on modal similarity generative adversarial networks(MS-SCH-GAN) is proposed.


## Research on Image Data Cleaning Based on ResNet 


Image data cleaning is an important research area. There is still significant research space in the field of image data cleaning. The Cleaning Model for Anomaly Image Data based on ResNet is proposed for anomaly image categories. The model enlarges the input data, uses improved ResNet to categorize and filter inaccurate data, then compares thresholds to eliminate the erroneous images. The proposed model is based on ResNet, which has been optimized and improved. 

