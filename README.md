# Lane Detection with Semantic Segmentation

## Methods

## Abstract

## Dataset(s)

## Plan of Action
1. Principles of Convolutional Neural Network
2. Image Segmentation
3. CNN Architectures


## 1. Principles of Convolutional Neural Network
Conceptually, a CNNs earlier layers extract **low level features such as edges**. Later layers use those lower level features to extract **higher level features such as shapes**. The image is fed in through one or more layers represented in this diagram as Conv 1 and Conv 2. Each layer contains multiple filters, which are represented in the diagram as the stacks of orange rectangles. **Each filter can extract features from the image.** And when those features are matched to labels, we then have the basis of a model that can classify a picture. There are often many filters in each layer, so at each layer, we pass the image through each filter. For example, if there are 64 filters in the first layer Conv 1, then effectively 64 filtered copies of the image have passed to the second layer. And if that had 64 filters, then 64 times 64 copies of the image have passed forward. 

![image](https://user-images.githubusercontent.com/59663734/135028355-56076b43-35fb-4aa7-8ac7-6718984e7544.png)

That can get computationally intensive. So pooling layers, which appear after Conv 1 and again after Conv 2 are used to reduce the number of computations. **Pooling is a methodology to reduce the number of pixels in the image while maintaining the features of the image and often enhancing those features**.

### 1.1 Convolutions
On the left side we have our input image of size ```5x5```. We assume we normalized our pixel values such that ```0``` represents ```black``` and ```1``` represents ```white```. We then have a convolutional ```filter``` of size ```3x3``` on the right also called a ```kernel```. We perform an ```element-wise matrix multiplication``` by sliding the kernel over our input image from the top left and sum the result. Our output will produce a ```feature map```. 

At a particular time ```t```, when our kernel is over our input image, the area where the convolution takes place is called the ```receptive field```. As our kernel is a ```3x3``` matrix, our receptive field is of similar dimension. The convoulution operation produces a scalar value into our future map. We perform the same element-wise matrix multiplication by sliding our filter to the right till we reach the bottom right of our input image. 

The operation can be understood by the animation below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145466555-bac3ed84-1c6a-4f0f-921c-d5aa2cd8d9e9.gif" />
</p>

The example here represents the image as 2D but our image will be 3D as we have our height, width and depth where the latter represents the ```3``` color channels of our image - ```RGB```. Since our image is 3D, our kernel also is 3D with its depth of the same dimension as the input image, that is, ```3```. We perform the same convolutional operation as on our 2D image and the matri multiplication will still result into a scalar. That is, our feature map will result into  ```n x n x 1``` dimension. Since we will have a number of filters passing onto our input image, our resulted feature map will be of size ```n x n x #filters```. What happens is that with each filter, we create a feature map and with a number of filters, the feature maps are stacked along the depth dimension.

For the example below, we perform convolution using a ```3x3``` filter and we produce a ```3x3``` feature map. In order for our Neural Nwtork(NN) to learn complex features we need to add ```non-linearity``` into our equation. Therefore, we pass our values of our feature map into a ```ReLu activation function``` such that the output values of our feature map are no longer a linear system but the relu function applied to them. The relu function is as follows:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145471102-b25dea62-8366-4ae0-8034-e6dd682caeb3.png" />
</p>
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145470766-33219f4f-370a-44c8-ad08-5a931d9f58bd.gif" />
</p>



We have several other activation function we can use however, we need to remember that it is paramount to add a ```differentiable non-linear activation function```  in order to update our parameters during backpropagation and for our NN to compute complex features. 
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/144385155-3ce66cf5-91c3-4ea8-8cd4-53dfde180b6f.png" />
</p>

For the example above, we see that our filter was able to detect the vertical line in our input image. Using the ReLu activation function, the values of the first and last column of our feature map turned to zero and only the middle column retained its values. This made our system powerful enough to detect that vertical line. But how do we know that this particular set of numbers of our filter will detect the vertical edge? The values of the kernel are ```learnable parameters``` by our NN. 

To sum up:

- Filters are used to find patterns in images
- The find these patterns by doing convolution, i.e, a matrix multiplication between the image and the filter
- Values of our filters or weights are learned by our NN through backpropagation

### 1.2 Strides

### 1.3 Padding

### 1.4 Pooling

### 1.5 Convolutions

### 1.6 Upsampling

### 1.7 Unpooling

### 1.8 Transposed Convolutions


## 2. Image Segmentation
Instead of locating an object within a rectangular bounding box, segmentation instead figures out the pixels that make up that object. In the image below, we have different objects(cars, humans, road,...) and instead of drawing bounding boxes, we've colored the image to denote each of the detected objects. We can then subdivide the image into segments, and these segments can help identify individual objects within the image. 

![image](https://user-images.githubusercontent.com/59663734/144401131-d1342cf6-46dc-4c42-bc5e-85875c446656.png)


There are two types of image segmentation, **semantic segmentation** and **instance segmentation.**

### 2.1 Semantic Segmentation

With semantic segmentation, all objects of the same type form a single classification. The image below has highlighted all vehicles as one item for example. The word semantic refers to **meaning** so all parts of the image that have the **same meaning**, and in this case all vehicles, are grouped into the same segment. 

![image](https://user-images.githubusercontent.com/59663734/144401257-c736ee05-fecb-499c-ae13-89454b6abb41.png)

In semantic segmentation, all objects of the same class are regarded as one segment. **Each pixel is usually associated with a class.** For example, all persons in an image are treated as one segment, cars as another segment and so on. Popular machine learning models that solve semantic segmentation are: **Fully Convolutional Neural Networks, U-net, DeepLab, ...**

### 2.2 Instance Segmentation
With instance segmentation, even objects of the same type are treated as different objects. We have seven distinct vehicles in the image below, and we've colored them differently to highlight this. You can think of each vehicle as a separate instance of a vehicle. 

![image](https://user-images.githubusercontent.com/59663734/144401435-8be660c0-f6e3-4a8b-b1a0-11c8eb6bb607.png)

For instance segmentation, each instance of a person is identified as a separate segment. **Multiple objects of the same class are regarded as separate segments.** though they all belong to the same class - Vehicle. One popular algorithm that solves instance segmentation is **Mask R-CNN.**


## 3. CNN Architectures

### 3.1 U-Net

### 3.2 Segnet

### 3.3 FCN

### 3.4 Deeplab

### 3.5 Mask R-CNN










