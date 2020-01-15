# Convolutional Networks

Este github presenta una serie de tutoriales para entender de la forma más clara posible (de acuerdo a mi percepción de la enseñanza) qué son las redes convolucionales y cómo funcionan.

Además, se facilitan "naive approaches" de las funciones de Convolucion2D y MaxPooling2D (escritas en numpy), las que puede revisar en `./utils/utils.py` para entender qué realiza cada operación. (*Las funciones no están optimizadas* )

Para lo anterior, he decidido separar este tutorial en capítulos (cada cuaderno de jupyter está enumerado).

# 1. **Convolution, Pooling and Padding**
  
  Veremos:
  - Qué es un tensor (estructura de datos usadas en las redes convolucionales, de ahí el nombre **Tensorflow**)
  - Operaciones de convolución, pooling y padding y cómo afectan a las imágenes sobre las cuales se le aplica cada uno. 
  - Terminología importante, como: kernel, stride. 

## 1.1 ¿Qué necesito?
  Trabajaremos con python. Utilizaremos las librerías [NumPy](https://numpy.org/) para procesamiento de datos y [OpenCV](https://pypi.org/project/opencv-python/) para cargar las imágenes (también puede usar skimage o keras). 
  
## 1.2 Links de interés

1. [KDDNuggets: Understanding deep convolutional neural networks](https://www.kdnuggets.com/2017/11/understanding-deep-convolutional-neural-networks-tensorflow-keras.html)
2. [Stanford CS class CS231n: Convolutional Neural Networks for Visual Recognition. ](http://cs231n.github.io/convolutional-networks/)
3. [How will channel RGB effect convolutional neural network? Santhiya Rajan's posts](https://www.researchgate.net/post/How_will_channels_RGB_effect_convolutional_neural_network)
4. [MEDIUM: Convolution operation of a cnn in 5 min](https://medium.com/@sushruth.konapur/convolution-operation-of-a-cnn-in-5-min-91757955835d)
5. [Kernel (image processing) - Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing))
