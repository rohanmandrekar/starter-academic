---
title: Cifar-10 Image Classifier
subtitle: Cifar-10 Image Classifier project

# Summary for listings and search engines
summary: Cifar-10 CNN implementation

# Link this post with a project
projects: []

# Date published
date: "2021-03-30T00:00:00Z"

# Date updated
lastmod: "2021-03-30T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Cifar-10 dataset https://pytorch.org/tutorials/_images/cifar10.png'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

tags:
- data mining

categories:
- blog
---

##  

{{< hl >}} _Switch to dark mode for better readability_ {{< /hl >}}

The goal for this project was to build an Image Classifier for the Cifar-10 dataset.

To begin with I started with [PyTorch's Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py) and then proceeded to make multiple changes to the model in an attempt to increase the test accuracy. I tried different things and came up with 7 different approaches(in addition to the existing default code) to tackle this problem.

[Here](https://github.com/rohanmandrekar/Cifar-10-/blob/master/best_attempt(model7).ipynb) is a link to the jupyter notebook with the best results

####Here is a graph comparing the Training accuracy for modles 4 - 8 :
training accuracy is missing for models 1-3 because I added the attribute in the 4th model

![png](./trainingacc.png)

####Here is a comparision of Test accuracy for each model :

![png](./testacc.png)

####Here is a plot showing the trend in loss for each model :

![png](./loss.png)

####This bar graph shows the accuracy of prediction for each class of image seperately :

![png](./bar.png)





