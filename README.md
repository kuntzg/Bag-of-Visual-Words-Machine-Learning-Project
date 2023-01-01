# EC503 Final Project
## Bag of Visual Words
##### Egor Badich, Zhengtao Wang, Gabrielle Kuntz, Suyang Yan, Chujun Qi


**Abstract**

&emsp; Bag of words is one of the most commonly used methods for categorizing objects. The main idea behind the bag of words is to generate visual words using K-means. Then extract key points into one of the visual words and construct a histogram based on the occurrences of the words. This report shows the whole process of image classification based on the bag of words model. All of the steps are implemented by matlab. 

**Introduction**

&emsp; In recent years, the bag of words model shows its power in image classification and text recognition. It includes different detectors and descriptors so that users can choose different ways to extract features from images. The basic idea of a bag of visual words model is to detect features from images and quantize them into “visual” words. Each part of the image that contains a feature would be assigned a vector to generate a description matrix. After generating the visual words, use histogram to count the number of visual words for each image and pick the word that appears the most frequently, which means the bag of words model will pick the word and assign it as the single vector for the image. 

&emsp; The process we go through to build our bag of visual words model is to build a feature extraction method, one k means algorithm, and a multiclass SVM classifier. The purpose of the feature extraction algorithm is to build visual words for each part of the input image. After getting all the visual words of one image, use k means algorithm to cluster different words. For instance, one image may contain different types of features, so they would obtain different descriptions. Thus, we need k means to separate them and then use histogram to find the exact word for the target image. After getting all the visual words of the training set and image set, use multiclass SVM to train the classifier and then match each image in the test set to a word.

**Conclusion**

&emsp; In one word, we implement the whole bag of words model using SIFT algorithm and k means algorithm. In order to test our result, we build a linear multiclass SVM to train and test our final approaches. The test accuracy of our bag of words model can reach 0.6667. 

![image](https://user-images.githubusercontent.com/75282197/210184510-14f380c6-1402-4f3c-8315-7e9b85ef43da.png)

Figure 9: Confusion matrix and accuracy we got from our bag of words model

&emsp; Based on our research, the common accuracy that a linear multiclass SVM can get is around 0.6. Nowadays, the bag of words model is still powerful in image classification, but it is not the best. Many algorithms, like Convolutional Neural Networks, can reach a high accuracy in image classification. Overall, the bag of visual words model has a high flexibility for customization and it is easy for us to understand. But it still has some limitations. For example, it does ignore spatial relationships. 

