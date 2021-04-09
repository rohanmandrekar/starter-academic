---
title: Overfitting using higher order linear regression
subtitle: The goal of this project is to learn about the concept of overfitting using the Higher order linear regression

# Summary for listings and search engines
summary: Overfitting using higher order linear regression ; linear regression using scikitlearn ; linear regression using scikit-learn 

# Link this post with a project
projects: []

# Date published
date: "2021-04-08T00:00:00Z"

# Date updated
lastmod: "2021-04-08T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: false

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'https://vitalflux.com/wp-content/uploads/2020/12/overfitting-and-underfitting-wrt-model-error-vs-complexity.png'
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

###What is overfitting?

Overfitting is a concept in data science, which occurs when a statistical model fits exactly against its training data. When this happens, the algorithm unfortunately cannot perform accurately against unseen data, defeating its purpose. [_Source_](https://www.ibm.com/cloud/learn/overfitting)

###How is overfitting different from underfitting?

If overtraining or model complexity results in overfitting, then a logical prevention response would be either to pause training process earlier, also known as, “early stopping” or to reduce complexity in the model by eliminating less relevant inputs. However, if you pause too early or exclude too many important features, you may encounter the opposite problem, and instead, you may underfit your model. Underfitting occurs when the model has not trained for enough time or the input variables are not significant enough to determine a meaningful relationship between the input and output variables.  [_Source_](https://www.ibm.com/cloud/learn/overfitting)

![jpg](./model-over-fitting.jpg)
[_Source_](https://www.ibm.com/cloud/learn/overfitting)

The goal of this project was to learn about the concept of overfitting using higher order linear regression.

Here is the link to the [Jupyter Notebook](https://github.com/rohanmandrekar/overfitting-using-higher-order-linear-regression/blob/main/overfitting.ipynb)





### Challenges faced:
Adding another convolutional layer for me was the biggest challenge. on my first attempt of adding another convolutional layer I simply changed the input and output parameters not realising that the required dimensions after flattening would also need to be changed. After facing errors after multiple trials i stumbled upon [Python Engineer's Tutorial on YouTube](https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=882s&ab_channel=PythonEngineer) which perfectly explained how the dimesions of the image changes after applying each convolutional and MaxPool layer. Below is a screenshot from his video that explains the formula used to find the actual dimensions of the image after passing through the convolutional layer.

![png](./ytss.png)


### My Observations:
On observing the bar chart below, you will notice that the model, even on it's best performance struggled to classify birds, cats, dogs, and deers. One reason for this that I could come up with was that cats, dogs, and deers being 4-legged animals might have been difficult to distinguish. In my opinion to tackle this problem the model should have been made a little more complex.
![png](./bar.png)

A trend that I observed was that the performance of the model increased when the number of nodes in the model were increased.

In the end this project was fun to do and was a great learning experience.





