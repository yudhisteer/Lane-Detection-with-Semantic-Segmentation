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






#### 1.7.1 Convolutions over Volumes




### 1.8 Batch Normalization



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










