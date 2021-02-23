# Capsule Networks

A Capsule Neural Network (CapsNet) is a machine learning system that is a type of artificial neural network (ANN) that can be used to better model hierarchical relationships. The approach is an attempt to more closely mimic biological neural organization.

The idea is to add structures called “capsules” to a convolutional neural network (CNN), and to reuse output from several of those capsules to form more stable (with respect to various perturbations) representations for higher capsules. The output is a vector consisting of the probability of an observation, and a pose for that observation. This vector is similar to what is done for example when doing classification with localization in CNNs.

Among other benefits, capsnets address the "Picasso problem" in image recognition: images that have all the right parts but that are not in the correct spatial relationship (e.g., in a "face", the positions of the mouth and one eye are switched). For image recognition, capsnets exploit the fact that while viewpoint changes have nonlinear effects at the pixel level, they have linear effects at the part/object level. This can be compared to inverting the rendering of an object of multiple parts. (Wikipedia article about "Capsule Networks": https://bit.ly/37ue7b2)

### Paper:
```
Dynamic Routing Between Capsules - Sara Sabour, Nicholas Frosst, Geoffrey E Hinton
Link: (https://arxiv.org/abs/1710.09829)
```

![alt text](https://repository-images.githubusercontent.com/168210197/14bde000-8c49-11e9-8f5c-798cf3b9eabb)
