---
title: Distracted Driver Detection
summary: Distracted Driver Detection using CNN on State Farm dataset
tags:
- CNN
- Machine Learning
date: "2021-12-04"

# Optional external URL for project (replaces project detail page).
external_link:

image:
  -caption: 
  focal_point: Smart

links:
- icon: github
  icon_pack: fab
  name: Github Rep
  url: https://github.com/rohanmandrekar/Distracted-Driver-Detection
url_code: ""
url_pdf: ""
url_slides: ""
url_video: ""



# Slides (optional).
#   Associate this project with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides = "example-slides"` references `content/slides/example-slides.md`.
#   Otherwise, set `slides = ""`.
#slides: example


---

{{< hl >}} _Switch to dark mode for better readability_ {{< /hl >}}

# Blog incomplete (work in progress)
# Blog incomplete (work in progress)
# Blog incomplete (work in progress)
# Blog incomplete (work in progress)

# Motivation(Why I chose this project):




## Features supported by the App:


## Explanation of a few algorithms:
### Conv2D:
2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

When using this layer as the first layer in a model, provide the keyword argument input_shape (tuple of integers, does not include the sample axis), e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

Below is a GIF showing the working of the Conv2D layer:

![gif](./conv2d.gif)

### MaxPool2D:
Max pooling operation for 2D spatial data.

Downsamples the input representation by taking the maximum value over the window defined by pool_size for each dimension along the features axis. The window is shifted by strides in each dimension. The resulting output when using "valid" padding option has a shape(number of rows or columns) of: output_shape = (input_shape - pool_size + 1) / strides)

The resulting output shape when using the "same" padding option is: output_shape = input_shape / strides

It returns a tensor of rank 4 representing the maximum pooled values. See above for output shape.

### BatchNormalization layer:
Layer that normalizes its inputs.

Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

Importantly, batch normalization works differently during training and during inference.

## My Contribution:
I designed my own classifier from scratch. below is the architecture for the CNN model:



## Challenges Faced:


## Experiments and findings:  



### References:


