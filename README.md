# Lane Detection with Semantic Segmentation

## Methods

## Abstract

## Dataset(s)

## Plan of Action
1. Principles of Convolutional Neural Network
2. Image Segmentation
3. CNN Architectures


## 1. Principles of Convolutional Neural Network

### 1.1 Convolutions
On the left side we have our input image of size ```5x5```. We assume we normalized our pixel values such that ```0``` represents ```black``` and ```1``` represents ```white```. We then have a convolutional ```filter``` of size ```3x3``` on the right also called a ```kernel```. We perform an ```element-wise matrix multiplication``` by sliding the kernel over our input image from the top left and sum the result. Our output will produce a ```feature map```. 

At a particular time ```t```, when our kernel is over our input image, the area where the convolution takes place is called the ```receptive field```. As our kernel is a ```3x3``` matrix, our receptive field is of similar dimension. The convoulution operation produces a scalar value into our future map. We perform the same element-wise matrix multiplication by sliding our filter to the right till we reach the bottom right of our input image. 

The operation can be understood by the animation below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145704667-8223d306-6c6c-49b1-958b-81df2cb3fca4.gif" />
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


### 1.2 Padding
For the earlier example, when we have a ```5x5``` input image and we do convolution with a ```3x3``` filter then the size of our feature map is reduced to ```3x3```. There are two downsides of the convolutional operation:

1. With each convolution our inputs will shrink constantly and our feature map will be too small to contain important features - ```shrinking output problem```.
2. The pixel of the top left of the input image is used only once by the filter whereas for the middle ones the filters have been overlapped during convolution. Therefore, we are throwing away a lot of the information in the edges of the image. 

To solve the above two problems, we need to pad the image by an additional border of ```1``` pixel on all 4 sides to preserve the size of our feature maps. So if we have a kernel of ```fxf``` dimension passed over an input image of ```nxn``` then our feature map would be ```n-f+1 x n-f+1```. With padding, our input image is now```7x7``` and after convolution the resulting feature map is ```5x5``` compared to ```3x3``` as earlier. The formula for the output image with padding ```p```, filter size ```f``` and dimension of input image ```n``` is ```n-f+1+2p x n-f+1+2p```.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145616316-4aedf02d-e93b-4018-946a-58d8edd83298.gif" />
</p>

But how do we know how much padding should we use?

#### 1.2.1 Valid Padding
It implies no padding at all - ```p = 0```. The input image is left in its valid/unaltered shape. The resulting feature map uses the formula ```n-f+1 x n-f+1```. 

#### 1.2.2 Same Padding
In this case, we add ```p``` padding layers such that the output image has the same dimensions as the input image. We know our feature map will be of size ```n-f+1+2p x n-f+1+2p```. To know the value of our padding we need to do the following operation: ```n-f+1+2p = n```. We make p the subject of formula and we get ```p = (f-1)/2```. We observe that the padding size depends only on the size of our kernel and not on our input image. So for a ```3x3``` kernel we using a padding = ```1``` and for a ```5x5``` kernel we use a padding = ```2```. From the paddding formula, it is better we keep out kernel size an odd number  since then it will be divisible by 2. 

To sum up:

 - With zero padding, every pixel of the image is visited the same number of times.
 - Padding gives similar importance to the edges and the center.

### 1.3 Strides
Stride specifies how much we move the convolution filter at each step. For the earlier example we have been using a stride of ```1``` where we moved our filter 1 pixel to the right and 1 pixel down until we reach the bottom right corner. We can use a stride ```2``` if we want less overlap with our receptive fields. However, this will make our receptive smaller skipping important locations. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145704758-4d606ac6-9a61-477f-8690-61340ed9e67a.gif" />
</p>

To rectify our formula, suppose we have an inout image of size ```n```, padding ```p```, filter size ```f``` and stride ```s``` then the dimension of the feature map is: ```((n+2p-f)/s)+1 x ((n+2p-f)/s)+1```.

To sum up:

- The large the strides are, the less of our image will be covered by the filter. However, the computation will be faster.
- Stride determines how the filter scans the image.

### 1.4 Pooling

Pooling is used to ```lower``` the dimension of the input images by taking the ```mean``` or finding the ```maximum``` value of different areas. For instance, a picture of a cat after a pooling layer will result in a blurry image with a lower size or lower resolution. The color palette and the color distribution over both images will be still very similar. The shapes on the blurry image will still resemble the original. However, it'll be much ```less expensive``` to do computations on this pooled layer than it is on this original image. Pooling is really just trying to ```distill``` that information.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145670204-c93125bc-8383-43df-8c83-b96572bf2689.png" />
</p>

With pooling, we reduce our number of parameters which result in **less training time** and **prevent overfitting**. For a 3D output, it downsamples each feature map independently by reducing the height and width but keeping the depth unchanged. 

We have 3 types of pooling: **Max-pooling, Average Pooling** and **Min-Pooling**.

#### 1.4.1 Max-Pooling
Suppose we have a window size of ```2x2``` and a ```4x4``` inout image. To compute max pooling, we will slide a window over the input, and simply takes the ```max value``` in the window. Then we keep doing that until we've covered the entire image so that at the end we have a ```2x2``` output. Min-pooling operation is similar to max-pooling except we take the ```minimum``` value.

Note that with this window size and stride number, it halved the size of the feature map. This is the main reason for using pooling: **downsampling the feature map while keeping the important information.**

#### But what is max-pooling really doing?
It is getting the most ```salient``` information from the image, which are these really ```high values```. This can be really important for distilling information where what we really care about is only the most **salient information**. For an intermediate layer, we are just extracting the ```weights``` with the highest values on this intermediate section.

#### 1.4.2 Average-Pooling
Average-pooling follows the same operation as max-pooling however, we take the average of the numbers in our sliding window. It is used less often than max-pooling.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145624189-d194ab55-8372-4b52-a87b-3d6a0e6943ce.png" />
</p>

One interesting property of pooling is that it has a set of hyperparameters, i.e, ```window size``` and ```stride number```, but it has **no** ```learnable parameters```. It is just a fized computation and gradient descent does not change anything. 

To sum up:

- In CNN architectures, pooling is typically performed with ```2x2``` windows, stride ```2``` and no padding. While convolution is done with ```3x3``` windows, stride ```1``` and with padding.
- Pooling reduces the height and width of our feature map but keeping the depth constant.

### 1.5 Upsampling
Up-sampling has the opposite effect of pooling. Given a ```lower``` resolution image, up-sampling has a goal of outputting one that has ```higher``` resolution although it may not result in a perfect image. The underlying principle is to infer values for the additional pixels.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145671294-c40214d2-a870-41de-842b-7f74a7b38851.png" />
</p>

There are several techniques of upsampling, namely: **Nearest-Neighbor Interpolation, Bilinear Interpolation, Bed of Nails, Max Unpooling** and **Transposed Convolutions**. 

#### 1.5.1 Nearest-Neighbor Interpolation
We first assign the value in the top left corner from the input to the top left pixel in the output. The other values from the input to pixels that are added distance of ```2``` pixels from that top left corner. There'll be exactly ```1``` pixel in between each of these, including a diagonal. Then assign the same value to every other pixel as it is to its ```nearest neighbor```. For every ```2x2``` corner in the output, the pixel values will look the same. You can also think of this as putting these values into the corners first and for every other pixel finding its nearest neighbor. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145676756-10812185-6a0a-4f9b-bdc1-f440a55fe8f7.png" />
</p>

#### 1.5.2 Bilinear Interpolation
In Bi-Linear Interpolation, we take the 4 nearest pixel value of the input pixel and perform a weighted average based on the distance of the four nearest cells smoothing the output. We perform linear interpolation in both direction - x and y axis. If we assume the distance between the ```20``` value pixel and the ```45``` value pixel is ```1``` then the distance between each subsequent pixel is ```1/3```. Hence, we calculate the missing pixel values using the equation below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145678789-50a514e8-cff2-408c-844a-0a562c5ef7f1.png" />
</p>

Below is an example of the upsampling result using the various techniques we have seen so far:
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145677050-66857075-d58f-41fc-8103-98d73b1008c2.png" />
</p>

#### 1.5.3 Bed of Nails
In Bed of Nails upsampling we first assign one pixel value in the top leaft corner and then leave a pixel distance between this pixel to assign the other pixels in our input image. The empty pixel in our output is filled with zeroes. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145680570-989ff12f-092d-488a-a16c-a2a1795d2cb4.png" />
</p>


#### 1.5.4 Max Unpooling
In max unpooling we remember the spatial information from max pooling. That is, when we take the max value of our sliding window on our input image with a stride ```2```, we preserve the index or the spatial information of that max value pixel. So in max unpooling, that max value is replaced back in our output at the same index of our original image before max pooling. Al the other pixels are filled with zeroes.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145680689-32c89687-6e66-472a-8381-c44ee13082e6.png" />
</p>


#### 1.5.5 Transposed Convolutions
The transposed convolutional layer, unlike the convolutional layer, is upsampling in nature. Transposed convolutions are usually used in auto-encoders and GANs, or generally any network that must reconstruct an image. We can use a ```2x2``` **learned** filter with stride equal to ```1```. We start by taking the top-left value from our input and get its product with every value in the ```2x2``` filter. That is, ```9``` in our input is multiplied by every number in our filter. The procedure is repeated by sliding the filter by one step and every pixel in our input is multiplied to the corresponding pixel index value in the filter. We then have 4 resulting maps which we will need to add to get our output. The addition is by simpling adding all pixel values of the same index. For example, to get the pixel value of 20 in our output we did: ```0 + 0 + 18 + 2``` where the first two zeroes are from the first and second feature maps respectively and 18 is from the third one and 2 is from the last one. 

With this computation, we can see that some of the values in the output are influenced much more heavily by the values from the input. For instance, the center pixel in the output is influenced by ```all``` the values in the input, while the corners are influenced by just ```one``` value.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145682257-19c09039-59e5-45c1-b095-cb93a62b36e7.png" />
</p>


An issue which arises with Transposed Convolution is the output to have a pattern resembling a ```checkerboard``` and this happens because when we upsample with a filter, some pixels are influenced much more heavily while the ones around it are not.


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145688277-5b38ef7e-9ac2-431d-b44d-cbfb92e33183.png" />
</p>

### 1.6 Dilated Convolution (Atrous Convolution)
Dilated convolutions are a way of increasing the receptive field of the network. These helps in understanding the overall picture rather than finer details. Thus, they cut the compute cost and provide a global view of the network in much lesser depth. Intuitively, dilated convolutions ```inflate``` the kernel by inserting spaces between the kernel elements. This additional parameter ```dilation rate``` indicates how much we want to widen the kernel. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145702734-f3734265-0802-4285-bac7-722a309b5aed.gif" />
</p>

 A ```3x3``` kernel with a dilation rate of ```2``` will have the same field of view as a ```5x5``` kernel, while only using ```9``` parameters. This delivers a wider field of view at the same computational cost. We use them if we need a wide field of view and cannot afford multiple convolutions or larger kernels.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145702888-01453567-025e-421a-ac6b-bea0b499af17.png" />
</p>

To sum up:

- Dilated convolution are very effective in segmentation tasks.
- A dilatation rate of 1 means normal convolution. 
- With a dilation rate > 1, we increase the receptive field which helps in memory and efficiency. 


### 1.7 CNN
A CNN model can be thought as a combination of two components: ```feature extraction``` part and the ```classification``` part. The convolution + pooling layers perform feature extraction. The fully connected layers then act as a classifier on top of these features, and assign a probability for the input image.

Conceptually, a CNN's earlier layers extract **low level features such as edges**. Later layers use those lower level features to extract **higher level features such as shapes**. The image is fed in through one or more layers represented in the diagram below as Conv 1 and Conv 2. Each layer contains multiple filters, which are represented as the stacks of orange rectangles. **Each filter can extract features from the image.** And when those features are matched to labels, we then have the basis of a model that can classify a picture. There are often many filters in each layer, so at each layer, we pass the image through each filter. For example, if there are 64 filters in the first layer Conv 1, then effectively 64 filtered copies of the image have passed to the second layer. And if that had 64 filters, then 64 times 64 copies of the image have passed forward. 

![image](https://user-images.githubusercontent.com/59663734/135028355-56076b43-35fb-4aa7-8ac7-6718984e7544.png)

That can get computationally intensive. So pooling layers, which appear after Conv 1 and again after Conv 2 are used to reduce the number of computations. Pooling, as explained above, is a methodology to reduce the number of pixels in the image while maintaining the features of the image and often enhancing those features. We repeat the procesdure of convolution and pooling several times depending on our architecture to extract the maximum and the most salient features in order to differentiate between our classes.

The result of our ```Feature Extraction layer``` will end up with a 3D feature map which we will then need to pass though a ```Fully Connected(FC) Layer``` also known as a ```Trained Classifier```. The Neural Network will learn on the values of our feature map obtained to match it with its corresponding label so as during inference it can make accurate prediction on unlabelled data. 

We will now dive deeping in each layer of a CNN.

#### 1.7.1 Convolutions over Volumes

Our imaged will have 3 channels - RGB - therefore our image will be a 3D volume and consequently our filter also be be a 3D volume. The convolution operation of the filter onto the input image will however produce a single scalar number in a 2D object. But since we will have a number of filters, we will gain output a 3D feature map. The height and width of the filter is not depending on the dimension of the input image but only the depth or the number channels of the input image must match the depth of the filter.

We have have different filters for each channel of our input image. For example we can have a vertical edge detector only for the red channel or we could have verticial edge detectors for all channels of our input image. For a ```3x3``` filter we have 27 learnable parameters and it we have 10 filters then we have ```270``` parameters plus a bias. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145707766-08119c98-f441-440e-bd5d-7c346779166a.gif" />
</p>

So we have a kernel for each channel and after convolution we produce 3 outputs as shown above. Then, these three outputs are summed together (element-wise addition) to form one single channel (3 x 3 x 1) as shown in yellow below. The yellow volume represents the feature map of one filter passing over an RGB image. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145708015-d8b31060-af8b-4917-97d5-53db580ff208.gif" />
</p>

In other words, we are sliding a smaller 3D volume(filter) over our 3D image. The 3D filter moves only in ```2-direction```, height & width of the image. That’s why such operation is called as ```2D``` convolution although a 3D filter is used to process 3D volumetric data.  Remember that the input layer and the filter have the same depth (channel number = kernel number). At each sliding position, we perform element-wise multiplication and addition, which results in a **single number**. The output is a one-layer matrix as shown below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145708292-05920816-ecaa-4fbf-a58c-7ff3f09ddfa1.png" />
</p>

Whatver the number of channel of our input, when applying a filter with the same channel number it would result into one output channel. So if we want our output feature map to have a ```D``` channels or a depth of ```D``` then we just need to have ```D``` number of filters. After applying ```D``` filters, we have ```D``` channels, which can then be stacked together to form the output layer.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145708451-45e87f58-0030-4719-874f-b516ee0224e4.png" />
</p>


#### 1.7.2 One Layer of CNN
When doing convolution we will output a 2D layer. And before stacking these layers, we will add a ```bias, b``` which is a Real Number and with broadcasting operation we will add this number to all the 27 parameters in our ```3x3``` filter. We will then perform a non-linearity activation function - ReLu -  of this 2D object to get a ```3x3``` output. It is these processed 2D outputs that we stack on each other to form our 3D feature map. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145709357-1a9dfa11-90e9-44d1-903b-1c896e800373.png" />
</p>

The values in our filters are the weights which the NN will learn. With the bias we have ```28``` parameters to learn in a ```3x3``` filter. It is important to note here that no matter how big our picture is, ```1000x1000``` or ```5000x5000```, the number of parameters will still remain ```28``` for one filter. The advantage of this is what makes CNN less prone to overfitting. Once we learn 10 feature detectors that work, we could apply this to very large images and the number of parameters will still remain fixed.

#### 1.7.3 Simple CNN Architecture
We will explore a simple CNN architecture similar to the LeNet-5 CNN architecture used on large scale to automatically classify hand-written digits on bank cheques in the United States.

Our architecture has two parts: ```feature extraction``` and ```classification```. In our Feature Extraction part we have two layers: **Layer 1** and **Layer 2**. In layer 1, we have our ```32x32``` image on which we apply ```6``` filters of size ```5x5``` with stide ```1```. This result in a ```28x28x6``` feature map. We then perform max-pooling with a window size of ```2x2``` and tride ```2``` which shrinks our height and width but the depth stays unchanged. We observe that as we go deeper in our CNN our height and width decreases but our depth, the number of channels, increases. 

In layer 2, we again perform a convolution and max-pooling with the same parameters which output a ```5x5x16``` feature map. We flatten our 3D feature map which contains ```400``` units(5x5x16). It is then passed to the Fully Connected layer of 120 units and again to another of ```84``` units to end in a softmax function with ```10``` units since we have 10 classes to predict.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145714004-217fc3e2-1cee-40bc-835d-63e9dd307080.png" />
</p>

Below shows the number of parameters and the activation size of the feature maps:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145714857-b5f0ac72-536f-4253-9ab2-8ba271e3ae82.png" />
</p>

We observe that the activation size decreases gradually and most of the parameters are found in the fully connected layer. Max-pooling does not have any learnable parameters and the convolutional layers have fewer parameters compated to the FC.

#### 1.7.4 Advantages of CNN
There are two advantages of using a convolutoonal layer before using a FC compared to only using a FC:

1. Parameter Sharing
2. Sparsity of Connections

##### 1. Parameter Sharing
From the table above, we find that the activation size of the input image is ```3072``` and suppose we needed to pass it into a layer with ```4704``` units then the total number of parameters to be learnt is 3072x4704 which is approx ```14 million```. And if we had larger images then our weight matrix will be increasingly large. However when using a ConvNet, we observe that the number of parameters is only 456 which is significantly smaller. The reason of the small number of parameters in a ConvNet is due to parameter sharing whereby a feature detector can be applied on different parts of an image. We don't really need a separate detector for the upper left corner of an image and another for the bottom right. We use the same detector all across the image during convolution and that greatly reduces our number of parameters.

##### 2. Sparsity of Connections
When performing convolution for a receptive field of a filter, the pixel values outside that window does not have any effect. In other words, the output value of an index in our feature map depends only on the comvoluton performed in that specific receptive field onto our image and the other pixels produces no interference. In that way, our ConvNet has less parameters which allows it to be trained with smaller training set and less prone to overfitting.   

Additionally, CNN are good at capturing ```translation invariance```. That is a picture of a dog shifted to the right will still be classfied as a dog even though it was not trained on such pictures. Convolutional structure help the NN encode that an image shifted will have similar features therefore be labeled similarly. 

### 1.8 Batch Normalization

The idea is that, instead of just normalizing the inputs to the network, we normalize the inputs to layers within the network. It’s called “batch” normalization because during training, we normalize each layer’s inputs by using the mean and variance of the values in the current mini-batch (usually zero mean and unit variance). Batch normalization optimizes network training. It has been shown to have several benefits:

- it stabilises optimisation allowing much higher learning rates and faster training
- it injects noise (through the batch statistics) improving generalisation
- it reduces sensitivity to weight initialisation
- it interacts with weight decay to control the learning rate dynamics

David C Page performed an experiment to demonstrate the effect of batch norm on optimisation stability. He trained a simple, 8-layer, unbranched conv net, with and without batch norm, on CIFAR10. The result is shown below:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145724597-77a0df6c-150e-42c1-b9df-9d9c2fe73831.png" />
</p>


It can be seen that the network with batch norm is stable over a much larger range of learning rates. The ability to use high learning rates allows training to proceed much more rapidly for the model with batch norm.

#### 1.8.1 Covariate Shift

To understand Batch Normalization, we can take the example of a single layer and a single activation unit with two input variables. We want to determine, based on the size and fur color of cats, if an image is a picture of a cat or not. 

We obeserve that the data of the size of cats is normally distributed around a midsize example with very few extremely small or extremely large examples. However, the distribution of fur color skews a little bit towards higher values with a much higher mean and lower standard deviation. The higher value can represent darker fur colors. It is important to note here that fur color and size does not have any correlation so we cannot really compare their values. 

The facts that we have two different distributions for our two variables impact the way our NN will learn. If it's trying to get to this local minimum and it has these very different distributions across inputs, this cost function will be ```elongated```. So changes to the weights relating to each of the inputs will have kind of a different effect of varying impact on this cost function. And this makes training fairly difficult, makes it slower and highly dependent on how the weights are initialized.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145762745-1363fa00-5ada-46ea-ae35-43e7a3aa513e.png" />
</p>

If a new training or test data has really light fur color, the state of distribution shifts or changes in some way. The form of the cost function would change too and the location of the minimum could also move. Even if the labels on our images of whether something is a cat or not has not changed. And this is known as ```covariate shift```. This happens often between training and test sets where precautions haven't been taken on how the data distribution is shifted. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145764110-114e2e9a-97b2-408b-9b07-1797a7ab3766.png" />
</p>


When we normalized our input variables such that the distribution of the new input variables <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;x_{1}^{'}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;x_{1}^{'}" title="\large x_{1}^{'}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;x_{2}^{'}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;x_{2}^{'}" title="\large x_{2}^{'}" /></a> will be much more similar with means equal to ```0``` and a standard deviation equal to ```1```. Then the cost function will also look **smoother** and more **balanced** across these two dimensions. And as a result training would actually be much easier and potentially much faster.

Also, no matter how much the distribution of the raw input variables change, from training to test, the mean and standard deviation of the normalized variables will be normalized to the same place around a mean of ```0``` and a standard deviation of ```1```. For the training data, this is done using the ```batch statistics```. As we train each ```batch``` we take the mean and standard deviation and we shift it to be around ```0```, and standard deviation of ```1```. And for the test data we can actually look at the statistics that were gathered overtime through the training set and use those to center the test data to be closer to the training data. And using normalization, the effect of this covariate shift will be reduced significantly.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145765276-73d46981-df81-40f8-b303-269ce6028929.png" />
</p>


#### 1.8.2 Internal Covariate Shift
The next dilemma we will face is experiencing covariate shift in internal layers of a Neural Network also known as ```Internal Covariate Shift```.

Let's examine the activation output of this second hidden layer of the neural network and look at the second node. When training the model, all the weights that affect the activation value are updated. And consequently, the distribution of values contained in that activation changes in our influence over this course of training. This makes the training process difficult due to the shifts similar to the input variable distribution shifts we saw earlier. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145771390-38e807a1-07cd-48bf-a2ca-5dde0b64ea6d.png" />
</p>

Batch normalization remedies the situation by normalizing all these internal nodes based on statistics calculated for each ```input batch``` in order to reduce the ```internal covariate shift```. And this has the added benefit of smoothing that cost function out and making the neural network easier to train and speeding up that whole training process.


To sum up:

- Covariate shift shouldn't be a problem if we make sure that the distribution of our data set is similar to the task we are modeling. That is, the test set is similar to our training site in terms of how it's distributed.
- Batch normalization smooth the cost function and reduces the effect of internal covariate shift. It's used to speed up and stabilize training.


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

## Conclusion

## References
1. https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
2. https://medium.com/@marsxiang/convolutions-transposed-and-deconvolution-6430c358a5b6
3. https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
4. https://medium.com/hitchhikers-guide-to-deep-learning/10-introduction-to-deep-learning-with-computer-vision-types-of-convolutions-atrous-convolutions-3cf142f77bc0
5. https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
6. https://towardsdatascience.com/batch-normalization-8a2e585775c9
7. https://e2eml.school/batch_normalization.html










