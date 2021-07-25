# Neural Networks Using TensorFlow
  
## Introducton:  
Neural Networks are a subset of Machine Learning, that uses interconnected layers of nodes, called neurons, to mimmick the working of the biological neurons. The idea of Neural Networks was inspired by the human brain and it forms the basis of Deep Learning. The basic architecture of a Neural Network consists of an input layer, one or more hidden layers, and an output layer.   
  
![image.png](attachment:6ddc153f-a1e3-434e-993c-05d26fe70121.png)
  
## Building Neural Networks in TensorFlow:  
To build and train Neural Networks with Tensorflow we use the [**keras module**](https://www.tensorflow.org/api_docs/python/tf/keras), which is an implementation of the high-level Keras API of TensorFlow. Keras is a deep learning API written in Python, running on top of the machine learning platform TensorFlow. It was developed with a focus on enabling fast experimentation. It is simple to use, flexible and powerful. Keras provides methods to prepare the data for processing, build and traain the model as well as methods to evaluate and fine tune the model parameters. 
  
![image.png](attachment:ed2822ce-3361-4cea-970b-b4573d6ea445.png)
  
### Defining the Architecture
A simple Neural Network can be built using the [Sequential class](https://www.tensorflow.org/guide/keras/sequential_model). A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor. Keras provides a wide range of layers that can be added to the model in it's [layers module](https://www.tensorflow.org/api_docs/python/tf/keras/layers). Layers can be added using the add() function or they can directly be defined in the object definition. The model is built using the compile() function.
  
### Training the model
The fit() method is used to train the model by passing the training data as arguments. Keras also keeps checkpoints during training and callbacks can be used to stop the training when specified requirements are met. Keras also keeps logs of the training process thus making it poassible to understand exactly what happens during the training process.  
  
### Evaluting the model
The performance of the model can be evaluated using the evaluate() method by passing the test data and labels as arguments. The metrics on which it has to be evaluated can be specified during the compilation. To get the predictions of the model the predict() method is used. After evaluation the model can be refined by tuning the hyperparameters. 

### Saving and Loading the model
The trained model can be saved using the save() method. Keras allows the model weights to be saved in several formats such as the HDF5 format and the SavedModel format. The saved model can then be loaded for use in other applications or it can be deplyed in a web service or an Edge device. 
  
Keras and TensorFlow provides an approachable, highly-productive interface for solving machine learning problems, with a focus on modern deep learning. Thus, TensorFlow is the go-to platform for researchers and engineers working with Neural Networks. 

