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
Before understanding Batch Norm, it is important we comprehend what is Normalization.

#### 1.8.1 Normalization
If we collect the data of all the activities of a single node in our layer for several iterations, we can actually constuct a distribution from that node's activities. The distribution need not be uniform and may not have a mean value of zero. Thefore, it is best if we normalize our distribution to make it as close to a Normal Distribution to have a mean of zero and a standard deviation of 1. The formula is:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146062008-fdbb2fde-b0c5-4423-aa42-466176ab343d.gif" />
</p>

It need not need to have a nice bell shape but atleast it will have a mean of zero and a standard deviation of 1.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146001319-6f70f9d9-4108-4372-b9ae-0c7324fc2564.png" />
</p>

#### 1.8.2 Importance of Batch Norm
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

#### 1.8.3 Covariate Shift

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


#### 1.8.4 Internal Covariate Shift
The next dilemma we will face is experiencing covariate shift in internal layers of a Neural Network also known as ```Internal Covariate Shift```.

Let's examine the activation output of ```layer 1``` of the neural network and look at the second node. When training the model, all the weights that affect the activation value are updated. And consequently, the distribution of values contained in that activation changes in our influence over this course of training. This makes the training process difficult due to the shifts similar to the input variable distribution shifts we saw earlier. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145937702-8974f522-192f-40e1-9b43-500ebf459122.png" />
</p>

Batch normalization remedies the situation by normalizing all these internal nodes based on statistics calculated for each ```input batch``` in order to reduce the ```internal covariate shift```. And this has the added benefit of smoothing that cost function out and making the neural network easier to train and speeding up that whole training process.

##### How does it work?
Suppose we set our batch number equal to ```32``` and we need to normalize the output of node 1 of layer 1 shown below. <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{z}_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{z}_{i}^{[l]}" title="\large {z}_{i}^{[l]}" /></a> represents the output from all the previous nodes.  Batch normalization considers every example <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{z}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{z}_{i}" title="\large {z}_{i}" /></a> in the batch. Since we have a batch of 32, we will have 32 <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{z}_{i}s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{z}_{i}s" title="\large {z}_{i}s" /></a> where i represents the <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;i^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;i^{th}" title="\large i^{th}" /></a> node and l represents the <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;l^{th}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;l^{th}" title="\large l^{th}" /></a> node.  From those 32 <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;z_{i}&space;s" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;z_{i}&space;s" title="\large z_{i} s" /></a>, we want to normalize it so that it has a mean of zero and a standard deviation of one. 



##### Batch Norm for Training
We start be calculating <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\mu&space;_{z_{i}^{[l]}}^{}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\mu&space;_{z_{i}^{[l]}}^{}" title="\large \mu _{z_{i}^{[l]}}^{}" /></a> which represents the mean of the batch of size 32. Then we also get the variance of the batch <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\sigma&space;_{z_{i}^{[l]}}^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\sigma&space;_{z_{i}^{[l]}}^{2}" title="\large \sigma _{z_{i}^{[l]}}^{2}" /></a> from these 32 values. To normalize these z values to a mean of zero and a standard deviation of one, we subtract the mean and we divide by the square root of the variance which is the standard deviation. We also add an <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\epsilon" title="\large \epsilon" /></a> here just to make sure that the denominator isn't zero. This is how we obtain <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\hat{z}_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\hat{z}_{i}^{[l]}" title="\large \hat{z}_{i}^{[l]}" /></a>.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145955178-edd9f9c8-47d6-4fff-ab2b-44819c6438ca.png" />
</p>

After we get the normalized value z-hat, we have parameters ```Beta```, which will be the ```shift factor``` and ```Gamma```, which will be the ```scale factor```, which are learned during training to ensure that the distribution to which we are transforming z is the optimal one for our task. After we completely normalize things to z-hat, we then rescale them based on these learned values, Gamma and Beta. This is the primary difference between **normalization of inputs** and **batch normalization**. With batch normalization we are not forcing the distribution to have zero mean and standard deviation of one every single time. It is after normalizing that we can go on and rescale things to an unnecessary task. Batch normalization gives us control over what that distribution will look like moving forward in the neural network. Within each batch, the activities of each element are separately shifted and scaled so that they have a zero mean and unit variance within the batch.
 <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{y}_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{y}_{i}^{[l]}" title="\large {y}_{i}^{[l]}" /></a> is what then goes into the activation function <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{a}_{i}^{[l]}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{a}_{i}^{[l]}" title="\large {a}_{i}^{[l]}" /></a>.

In summary: 

_Batch normalization is an element-by-element shift (adding a constant) and scaling (multiplying by a constant) so that the mean of each element's values is zero and the variance of each element's values is one within a batch. It's typically inserted before the nonlinearity layer in a neural network._

##### Batch Norm for Testing
During testing, we want to prevent different batches from getting different means and standard deviations because that can mean the same example, but in a different batch would yield different results because it was normalized differently due to the specific batch mean or specific batch standard deviation. Instead, we want to have stable predictions during test time. During testing, we use the ```running mean``` and ```standard deviation``` that was computed over the **entire training set**, and these values are now fixed after training.

The expected value, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;E(z_{i}^{[l]})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;E(z_{i}^{[l]})" title="\large E(z_{i}^{[l]})" /></a>, of those z values is the running mean and the square root of the variance of those z values, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;{\sqrt{Var(z_{i}^{[l]})}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;{\sqrt{Var(z_{i}^{[l]})}}" title="\large {\sqrt{Var(z_{i}^{[l]})}}" /></a>, is the standard deviation. We still have that Epsilon, <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\epsilon" title="\large \epsilon" /></a>, to prevent the denominator from going to zero. After that, we just follow the same process as we did in training and we feed these normalized values into the learn parameters and then the activation function. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/145992811-4c86a046-3149-4463-954d-6b97d4d886cd.gif" />
</p>

To sum up:

- Covariate shift shouldn't be a problem if we make sure that the distribution of our data set is similar to the task we are modeling. That is, the test set is similar to our training site in terms of how it's distributed.
- Batch normalization smooth the cost function and reduces the effect of internal covariate shift. It's used to speed up and stabilize training.
- Batch norm introdues learnable shift and scale factors.
- During tests, the running statistics from training are used.
- In TensorFlow and PyTorch all we have to do is create a layer called batch norm, and then when our model is put into the test mode, the running statistics will be computed over the whole data set for us.


## 2. Image Segmentation
The goal of image segmentation is to recognize areas in an image. It does this by assigning every pixel to a particular class with pixels that are in the same class having similar characteristics. Image segmentation can be described in these three steps:

1.  It will first determine the shape of each object, not with bounding boxes, but the outline of the shape will be determined. 
2.  We then partition an image into multiple segments and have each associated with an object. 
3.  We'll get each pixel in the image and have it classified into a different class. 

Instead of locating an object within a rectangular bounding box, segmentation instead figures out the pixels that make up that object. In the image below, we have different objects(cars, humans, road,...) and instead of drawing bounding boxes, we've colored the image to denote each of the detected objects. We can then subdivide the image into segments, and these segments can help identify individual objects within the image. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146134489-02844d4a-f91c-4abd-a27d-eea22fd901cd.png" />
</p>

Below is an example of a ```pixel map``` created for two classes. On the left we have the image - the data, and on the right we have the the label where it is labelled which pixels are people and which ones are background.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146132903-8b252bae-4e1a-4adc-ac24-c1467ba075b3.png" />
</p>
<p align="center">
  Classes Indices: [0 = Background, 1 = People]
</p>

### 2.1 Image Segmentation Basic Architecture
 The high level architecture for an image segmentation algorithm consists of an ```encoder``` and a ```decoder```.
 
 #### 1. Input Image
 We start with our colored input image of size ```224 x 224 x 3``` where ```3``` represents the three RGB channel. The image is then process with an encoder. 
 
 #### 2. Encoder
An encoder is a ```feature extractor```. One common method to extract features is one which we explored before - ```CNN```. Through a series of convolutio nand pooling, the CNN extract features and return a feature map. The **earlier layers** extract **low level features** such as **lines** and these lower level features are successfully aggregated into **higher level features** such as **eyes** and **ears**. The aggregation of successful higher level features is done with the help of ```downsampling``` or ```pooling```. That is, the CNN will represent an image with fewer pixels.
 
 
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146135687-21045da4-c7ef-43bc-ab4c-759ac0f94550.png" />
</p>

 
  #### 3. Decoder
  Whereas before in our CNN we had a Fully Connected layers to do a classification based on the learned filters, the image segmentation architecture will take the ```downsampled feature map``` and feed it to a decoder. The decoder's task, which is also a CNN, is to take the features that were extracted by the encoder and work on producing the models output or prediction. The decoder assigns **intermediate class labels** to each pixel of the feature map, and then **upsamples** the image to slowly add back the fine grained details of the original image. The decoder then assigns more fine grained intermediate class labels to the upsamples pixels and repeats this process until the images upsampled back to its original input dimensions. The final predicted image also has the final class labels assigned to each pixel. This then gives a **pixel wise labeled map**.

  #### 4.  Pixel-wise Labeled Map
   The pixel wise labeled map will be the size of the original image - 224 x 224 - with the **third dimension** being the **number of classes**. 
 
To sum up:
- Encoder is a CNN without its Fully Connected layers.
- Aggregates low level features to high level features.
- Decoded replaces the Fully Connected layers in a CNN.
- We upsample image to the original image to generate a pixel mask.

There are two types of image segmentation, **semantic segmentation** and **instance segmentation.**

### 2.2. Semantic Segmentation

With semantic segmentation, all objects of the same type form a single classification. The image below has highlighted all vehicles as one item. The word semantic refers to **meaning** so all parts of the image that have the **same meaning**, and in this case all vehicles, are grouped into the same segment. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/144401257-c736ee05-fecb-499c-ae13-89454b6abb41.png" />
</p>

In semantic segmentation, all objects of the same class are regarded as one segment. **Each pixel is usually associated with a class.** For example, all persons in an image are treated as one segment, cars as another segment and so on. Popular machine learning models that solve semantic segmentation are: **Fully Convolutional Neural Networks, U-net, DeepLab, ...**

### 2.3. Instance Segmentation
With instance segmentation, even objects of the same type are treated as different objects. We have seven distinct vehicles in the image below, and we've colored them differently to highlight this. You can think of each vehicle as a separate instance of a vehicle. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/144401435-8be660c0-f6e3-4a8b-b1a0-11c8eb6bb607.png" />
</p>

For instance segmentation, each instance of a person is identified as a separate segment. **Multiple objects of the same class are regarded as separate segments.** though they all belong to the same class - Vehicle. One popular algorithm that solves instance segmentation is **Mask R-CNN.**


## 3. CNN Architectures For Segmentation
Fully Convolutional Neural Networks(FCN) was first proposed for segmentation. Then there are various networks based on these fully convolutional networks.

Fully Convolutional Neural Network:
- SegNet
- UNet
- PSPNet
- Mask-RCNN

### 3.1 Fully Convolutional Neural Network(FCN)
The goal of the FCN was to replace the fully-connected layers and typical CNNs with **convolutional layers** that act as the **decoder**. The encoder layers detect features and downscale the image, and the decoder layers upscale the image and create a pixel wise labeled map.

Filters are learned in the usual way through **forward inference** and **backpropagation**. As the image passes through convolutional layers, it gets downsampled. Then the output is passed to the decoder section of the model, which are additional convolutional layers. At the end is a ```pixel-wise prediction layer``` that will create the ```segmentation map```. FCN's encoders are ```feature extractors``` like the feature extracting layers using object detection models such as ```VGG16```, ```ResNet 50```, or ```MobileNet``` which have pre-trained feature extraction layers that we can use.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146221061-f750262a-7605-43fb-a0f4-1f815005683a.png" />
</p>

The decoder part of the FCN is usually called ```FCN-32```, ```FCN-16``` or ```FCN-8``` where the number in the title represents the number of stride size during upsampling.  The smaller the stride, the more detailed the processing.  The decoder layers upsamples in the image step-by-step to its original dimensions so that we get a pixelwise labeling, also called ```pixel mask``` or ```segmentation mask``` of the original image.  From the result below, the resolution improves as the stride decreases and at stride  = ```8``` is closest to the ground truth.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146316209-7035cfc0-8984-431a-9073-15b7f50c87dd.png" />
</p>

Note: A pooling of window size ```2x2``` with stride ```2``` reduces the height and width of the input by half. But the depth remains constant. An input image of size ```256x256```  would get pooled to ```128x128``` and so on. 

#### 3.1.1 FCN-32
The architecture has five pooling layers. Each pooled result gets its dimensions reduced by half, **five times**. The original image gets reduced by a factor of <a href="https://www.codecogs.com/eqnedit.php?latex=2^{5}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{5}" title="2^{5}" /></a> which equals ```32```. To upsample the final pooling layer back to the original image size, it needs to be upsampled by a factor of ```32```. This is done by upsampling with a stride size of ```32```, which means that each input pixel from ```Pool 5``` is turned into a ```32x32``` pixel output. This ```32``` times upsampling is also the pixelwise prediction of classes for the original image.

#### 3.1.2 FCN-16
FCN-16 works similarly to FCN-32, but in addition to using Pool 5, it also uses Pool 4. In step 1, the output ofPpool 5 is upsampled by a factor of ```2```, so the result has the same height and width as Pool 4. Separately, we use the output of Pool 4 to make a **pixelwise prediction** using a ```1x1``` convolution layer. The Pool 4 prediction is added to the 2x upsampled output of Pool 5. The output of this addition is then upsampled by a factor of ```16``` to get the final pixelwise segmentation map. Upsampling with a stride of ```16``` takes each input pixel and outputs a ```16x16``` grid of pixels.


#### 3.1.3 FCN-8
FCN-8 decoder works very similar with the same first two steps, but instead of upsampling the summation of the Pool 4 and 5 predictions by ```16```, it will 2x upsample it, and then add that to the Pool 3 prediction. This is then upsampled by ```8```. 

Below is the architecture of the FCN model:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146318557-8f02c9e2-49a4-406e-bd45-45cab1ca26b9.png" />
</p>

### 3.2 SegNet
SegNet is very similar to FCN with a notable optimization which is the ```encoder layers``` are **symmetric** with the ```decoder layers```. They are like mirror images of each other with the same number of layers and the same arrangement of those layers. That is, for each pooling layer that downsampled in the encoder there is an upsampling layer in the decoder section. The first segment has two convolutional layers, followed by a pooling layer. The last segment is a mirror image of this with an upsampling layer followed by two convolutional layers. The same symmetry is found in the second layer and the second-to-last one, and so on for the rest of the image.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146326119-d9d4ba96-ab20-4986-a8fc-0a07263449d2.png" />
</p>

The difference between SegNet and FCN is that it has **Skip Connections** and **Batch Normalization**.

### 3.3 U-Net
U-Net was proposed in 2015 in a paper by the name of _U-Net: convolutional networks for biomedical image segmentation_ and was written by Olaf Ronneberger, Philip Fischer and Thomas Brox. 

U-Net is also a fully convolutional neural network, but with the key difference that in addition to the upsampling path, ```skip connections``` between the encoder and the decoder are also used. These skip connections are denoted by the curved arrows that reverse the U shape. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146351786-b63678f9-f990-4797-900a-eb2e6d14fee0.png" />
</p>

In general the U-Net is divided into three parts: **Encoder, Decoder** and **Bottle-neck**.

#### 3.3.1 Encoder
On the left of the U-shape we have the Encoder which is similar to the FCNs. We have an image fed into convolution layers and then down sampled using max pooling.

- **1st level**: if the image is fed in as a ```128x128``` and pass through two convolutional layers that have ```64``` filters each, the subsequent images when pooled, will be ```64x64```. This happens because a max pooling layer with a ```2x2``` window and a stride of ```2``` will reduce the dimensionality by half. 

- **2nd level**: the images are passed through two layers of convolutions with ```128``` filters each and they are then pooled from ```64x64``` to ```32x32```. 

- **3rd level**: the ```32x32``` matrices are passed through two layers of ```256``` filters each and then pooled to ```16x16```. 

- **4th level**: the ```16x16``` images are fed through two layers of ```512``` filters each, before being pooled into an ```8x8``` at the 5th level.

Thus from the 1st level to the 5th, a ```128x128``` image is filtered and down sampled into ```8x8``` blocks.

#### 3.3.2 Bottleneck
The bottleneck is an additional element in the unit architecture which is a simple convolutional layer with ```1024``` filters. It can further extract features, but it doesn't have a pooling layer to follow it.


#### 3.3.3 Decoder
At the 5th level we upsample our ```8x8``` block to ```16x16``` and move to the 4th level of the U-shape of the decoder. 

- **4th level**: we take the ```512``` filters from the layer of the encoder that's at the same level as this decoder layer. Since the encoder layer and decoder layer are at the same level in the unit, they also have the same height and width of ```16x16``` and they also have the same number of filters at ```512``` each. So we will **concatenate the filters from the encoder with the filters of the decoder** for a total of ```1024``` filters. We then pass this concatenated set of ```1024``` filters through ```2``` convolutional layers.And this pattern continues through the decoder. We'll upsample the blocks to ```32x32``` and move up to the 3rd level.

- **3rd level**: We'll take the filters from the encoder on the same level, concatenate them to the blocks from the decoder and pass the entire thing through ```2``` convolutional layers. So  we upsampled to ```64x64``` and move up to the 2nd level. 

- **2nd level**: We again combine the filters from the encoder with the decoder and pass them through two convolution layers and finally, upsample to ```128x128``` and move up to 1st level.

- **1st level**: We concatenate the filters from the encoder and decoder and pass them through the two convolution layers.

Finally, the output segmentation map is obtained, by performing ```1x1``` convolution with the filters equal to the number of classes on the output of the final stage and the upsampling path. Recall a ```1x1``` filter is the same as taking all the numbers in our feature map and multiplying each of these numbers in one slice in the same position and width by 192 weights(for example below), apply a non-linearity to it and output it into a feature map. 


<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146508017-c3d9619e-9809-462f-8a6e-0646b358a0c5.png" />
</p>


### 3.4 Region-Based Convolutional Neural Network (R-CNN)
R-CNN is a method of using a region based CNN to implement ```selective search``` algorithm to extract the top ```2000``` region proposals among millions of regions of interest (ROI) proposals from an image and feed it to a CNN model. R-CNN was proposed in the paper: _Rich feature hierarchies for accurate object detection and semantic segmentation_ by Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik in 2013. 

In order to perform **object detection**, we need to first peform **object localisation**. That is, we first ask: "Where is the object in the image?" and only then ask: "What is that object?". One approach discussed in the project [Automating Attendance System using Face Recognition with Masks](https://github.com/yudhisteer/Face-Recognition-with-Masks) is to use a trained classfier and run a sliding window all thoughout the image with a spesific stride number until it localizes the object. This approach is called ```Exhaustive Search```. One major drawback of this approach is that it classifies a lot of region where there is no object. For example, in the example below on the left, our sliding window in red is classifying a part of the image where it is basically blank - no interesting object. A solution to this is to run a segmentation algorithm to the image and find blobs for region of interest. We put these blobs in boudning boxes and then run a classfiier to detect the object. For the image below on the right, the blue blob would predict a car and the yellow ones would predict traffic signs. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146538708-eac37428-9d5e-4538-85e1-0ba9cc1ce221.png" />
</p>

The R-CNN consists of 3 main components: **Selective Search**, **Feature Extraction** and **Prediction**.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146641842-91c4e192-3c84-4663-a518-f7bd389806df.png" />
</p>


#### 3.4.1 Selective Search
The selective serach take as input an image. It then extracts ```region proposals```. It generates initial ```sub-segmentations``` to create multiple regions in the image. Next, we recursively combine the similar smaller regions into larger ones. We combine these regions based on **color similarity, texture similarity, size similarity**, and **shape compatibility**. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146642226-252df47c-aac1-4e9a-99ec-8b83ac33224a.png" />
</p>

We can output the candidate object boudning boxes for each semantic region. As seen in the image below, the initial segmentation is noisy and hence, it has a lot of bounding boxes. As we combine the regions together(around ```2000``` proposed regions), we reach an ideal number of semantic regions and crop the bounding boxes for each region. Our next step would be the feature extraction.

#### 3.4.2 Feature Extraction
In the feature extraction step, the model extracts features from each of these ```2000``` region proposals using a pre-trained convolutional neural network such as the ```AlexNet``` architecture. In order to adjust each region proposal to fit as the input to the R-CNN, we ```warp``` the image dimensions to create ```warped regions``` to fit the AlexNet input dimensions.


#### 3.4.3 Prediction
The prediction step uses ```Support Vector machines(SVM)``` as opposed to dense layers to classify the class of each proposals. Secondly, we train a linear regression model for bounding box prediction for each proposed region. 

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146643833-9cb3aebb-5674-4f94-936a-96e5a64a6c5d.png" />
</p>

**Note:** ```Transfer learning``` is used to pre-train the CNN section of the R-CNN model, and then fine tune that model to this specific task. Researchers identified a large **auxilary dataset** for pre-training. The pre-training auxiliary task is helping the model to perform its **domain specific task** better. Note that the images in the auxiliary dataset are not warped, like the region proposals used in this object detection task. These images also don't have bounding box labels, but the auxiliary data can still be used for pre-training, even when it's different than the data that's used for the desired domain specific task. The auxiliary data can help the model learn generally useful feature extraction, if it's a large dataset. So pre-training is normally performed on very large datasets, even if there are different classes or different formats than the actual final task that we want to perform.

#### R-CNN Issues

-  Finding the areas using selective search could be very slow and running each of the areas of interest, up to 2,000 of them, through the CNN could also be **slow** and **computationally expensive**. R-CNN framework lacked speed and end-to-end trainability.
-  Another large major problem was the **memory** requirement because of the need to have multiple classifiers for each class of objects.

We conclude that R-CNN is slow in both training and prediction. ```Fast R-CNN``` was then developed to overcome this bottleneck of R-CNN.

#### 3.4.4 Fast R-CNN
 Ross Girshick proposed an updated architecture called ```Fast R-CNN``` to improve the speed and memory issues of R-CNN.
 
<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146649910-42df4c48-29f9-4607-8783-f23f92bdc682.png" />
</p>

1.  An image is fed to the network along with a set of region proposals. Note that the difference between Fast R-CNN and the original R-CNN is that instead of using Selective Search to generate several region proposals for each image, Fast R-CNN expects these region proposals as inputs and it doesn't generate them itself.

2.  The CNN processes the image and outputs a set of features into a ```feature map```. Remember that the detected features are stored relative to where they're detected in the original image. For instance, if it's a heart and a horse's mouth that are detected, the heart feature are stored near the top of the map and the horse's mouth is stored near the bottom, similar to where they were in the original image.

3.  We then use the input region proposals and extract the ```region of interest(ROI)``` from this feature map and create a ```region proposal feature map```, one for each proposed region. This process is called the ```region of interest projection```. The difference between the first feature map and this region proposal feature map is that the original feature map is for the **entire image**, whereas the region proposal feature map is for the **specific** subsections of the proposed region of the image.

4. We will then ```downsample(pooling)``` this feature map with the help of a region of interest pooling layer to get a fixed length feature map of a consistent height and width. This means that regardless of the dimensions of each proposed region, which may vary in size, the fixed length feature map is consistently the same size.

5. To make this map usable, we can then ```flatten``` the fixed size feature map into a ```one-dimensional``` vector, which we'll call the ```region of interest feature vector```, and we will create this region of interest feature vector using a few ```fully connected layers```.

6. We can then use this region of interest feature vector to generate two outputs:
- The first output uses a ```fully-connected layer```, followed by ```Softmax``` in order to classify the image. 
- The second output uses a ```fully-connected layer``` and ```linear regression``` outputs to define the **size** and **location** of the bounding box for that classified object. 


In summary, ```no selective search``` is done to find the regions of interest. Instead a ```ConvNet``` is used and its filters determine those ```regions of interests```. The **entire image**, and not subsets, is passed into the convNet.  This saves the expensive Selective Search process. The ConvNet trained on finding features can then give us a ```feature map``` of the image.  Note here that we are now generating region proposals based on the last feature map of the network, not from the original image itself. As a result, we can train just one CNN for the entire image. The feature map outputs can then be ```pooled``` and fed through a ```fully-connected dense layer``` to get a feature vector representing our ```regions of interest``` within the image. The feature vector can then be classified through a fully connected layer with ```Softmax``` to get our ```classification```, and another with ```Regression``` to get our ```bounding boxes```.

#### Fast R-CNN Issues

- Fast R-CNN is much faster in both training and testing time. However, the improvement is not dramatic because the region proposals are generated separately by another model and that is very expensive.
- It takes around ```2``` seconds per image to detect objects which can still be considered slow if we have a large dataset.


#### 3.4.6 Faster R-CNN
The main goal of Faster R-CNN was to replace the slow selective search algorithm with a fast neural net hence, introducing the **learnable** ```region proposal network*RPN)``` to propose regions of interest in the region proposal feature map.

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146651868-2abda632-5ba3-464a-8a76-493b97c6aebf.png" />
</p>


RPN is a ```fully convolutional network(FCN)``` which, as discussed above, uses convolutions are not dense layers. So we can simultaneously predict object bounds and object scores for each pixel.  The RPN is designed to be trained end-to-end to generate **high quality region proposals**, which are used by faster R-CNN for detection. The RPN makes object proposals possible employing **anchors** or priors.

1.  Faster R-CNN the entire image is passed into a ConvNet. 
2.  This feature map then has a ```3x3``` sliding window that goes across it to find areas of interest and maps it to a lower dimension. A new entity called a **region proposal network(RPN)** is used create anchor boxes on the image.  
3.  The center of the anchor box comes from the coordinates of the sliding window and the boundaries of the box come from the RPN, giving us a score that the boundaries of the box better fit the objects. That is, we look at each location in our last feature map and consider ```k``` different boxes centered around it: a tall box, a wide box, a large box and so on. For each of those boxes, we output whether or not we think it contains an object, and what the coordinates for that box are. 
4.  Although the RPN outputs bounding box coordinates, it does not try to classify any potential objects: its sole job is still ```proposing object regions```. If an anchor box has an **“objectness”** score above a certain threshold, that box’s coordinates get passed forward as a ```region proposal```.
5.  We then train a Fast R-CNN object detection model using the proposals generated by the current RPN. We add a pooling layer, some fully-connected layers, and finally a softmax classification layer and bounding box regressor.

In summary, **Faster R-CNN = RPN + Fast R-CNN**

Although Faster R-CNN is complex, its core design is the same as the original R-CNN: **hypothesize object regions and then classify them**..

Below is a summary of the 3 types of R-CNN:

<p align="center">
  <img src= "https://user-images.githubusercontent.com/59663734/146651712-adecaa8c-6aa6-4906-bbe1-7cbd3a5a617a.png" />
</p>





#### 3.4.7 Mask R-CNN




To sum up:

- **R-CNN**: Propose regions. Classify porposed regions one at a time  and output label with SVM and bounding box with regression. 
- **Fast R-CNN**: Performing feature extraction over the image **before** proposing regions, thus only running one CNN over the entire image instead of 2000 CNN’s over 2000 overlapping regions. Replacing the SVMs with a softmax layer, thus extending the neural network for predictions instead of creating a new model.
- **Faster R-CNN**: Use convolutional network to propose regions. 

## Conclusion

## References
1. https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
2. https://medium.com/@marsxiang/convolutions-transposed-and-deconvolution-6430c358a5b6
3. https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
4. https://medium.com/hitchhikers-guide-to-deep-learning/10-introduction-to-deep-learning-with-computer-vision-types-of-convolutions-atrous-convolutions-3cf142f77bc0
5. https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d
6. https://towardsdatascience.com/batch-normalization-8a2e585775c9
7. https://e2eml.school/batch_normalization.html
8. https://github.com/mrgloom/awesome-semantic-segmentation
9. https://www.topbots.com/semantic-segmentation-guide/
10. https://medium.com/beyondminds/a-simple-guide-to-semantic-segmentation-effcf83e7e54
11. https://www.geeksforgeeks.org/selective-search-for-object-detection-r-cnn/
12. https://laptrinhx.com/object-detection-algorithms-r-cnn-vs-fast-r-cnn-vs-faster-r-cnn-1543446592/
13. https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html
14. https://www.youtube.com/watch?v=Z9nCBtaEb_g
15. https://nanonets.com/blog/semantic-image-segmentation-2020/
16. https://towardsdatascience.com/deep-learning-for-object-detection-a-comprehensive-review-73930816d8d9










