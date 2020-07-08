# Using Naive Bayes Classification to build a SPAM Filter


## Introduction

In this project, we built a spam filter software to classify messages as SPAM or HAM (not spam). To train the algorithm, we used the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). The dataset was collected by Tiago A. Almeida and José María Gómez Hidalgo. Our goal was to design a SPAM filter with an accuracy of over 80% - which means we expect over 80% of the messages to classify correctly as spam or ham.

## Data Overview

The data set `SMSSpamCollection` is a text file of 5,574 SMS messages in English. The data has already been classified as `ham` (legitimate) or `spam`. It was downloaded using pandas `read_csv` function. To download the file correctly, the parameters used were - `sep='\t'` and `header=None`. We also defined the column names as `Label` and `SMS` in the function. Below is a snapshot of the first five rows - 

|   | Label | SMS                                               |
|---|-------|---------------------------------------------------|
| 0 | ham   | Go until jurong point, crazy.. Available only ... |
| 1 | ham   | Ok lar... Joking wif u oni...                     |
| 2 | spam  | Free entry in 2 a wkly comp to win FA Cup fina... |
| 3 | ham   | U dun say so early hor... U c already then say... |
| 4 | ham   | Nah I don't think he goes to usf, he lives aro... |

87% of the dataset is labeled as `ham` while only 13% is `spam`. The sample looks representative, since most messages that people receive day to day are ham. 

| ![Image 1](Images/Spam%20vs%20Ham%20in%20dataset.png) |
| :--: | 


| Label | ValueCount |
|-------|------------|
| ham   | 0.865937   |
| spam  | 0.134063   |


## Training & Test Set 


To start, we randomize and split the dataframe into a training and test set. The train set will *train* the computer to classify the messages. The test set will *test* the accuracy of our spam filter. We use 80% of the dataset for training as we want to be able to train the algorithm on as much data as possible. After splitting, the train set has 4458 SMS messages while the test set has 1114 messages. Finally, we analyze the percentage of spam and ham messages in the train and test set. The idea is to have the same percentage of spam vs ham messages as in the original dataset. 


| ![Image2](Images/Spam%20vs%20Ham%20in%20train%20data.png) | ![Image3](Images/Spam%20vs%20Ham%20in%20test%20data.png) |
| :--: | :--: | 


### What is Naive Bayes algorithm?

Naive Bayes is a powerful algorithm for supervised pattern classification. It is based on Bayesian classification theory with the "*naive*" assumption that the features in the dataset are mutually independent (i.e., the occurence of one does not change the probability of the other). Given two events Y & X, Bayes theorem calculates conditional probability as - 

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28Y%7CX%29%3D%5Cfrac%7B%5CPr%28X%7CY%29%5CPr%28Y%29%7D%7B%5CPr%28X%29%7D)

- Pr(Y|X) is the probability of Y to occur if X already occured, also known as the posterior probability
- Pr(Y), Pr(X) - probability of Y, X to occur, also known as prior probability
- Pr(X|Y) - probability of X to occur given Y 

A good example to consider would be HIV testing. The probability that a given person has HIV (Pr(HIV<sup>+</sup>)) is the prior (before testing) probability. Whereas, the probability that a person has HIV after testing positive (Pr(HIV<sup>+</sup>|T<sup>+</sup>))  is the posterior probability (after testing). In an ideal world, diagnostics test would be 100% reliable and the Pr(HIV<sup>+</sup>|T<sup>+</sup>) would always be 1. But, in real life there is a small chance (testing error, human error etc.) that you don't have HIV even though you test positive. This is known as a *false* positive. 

In the broader context of classification, Bayes theorem can tell us the probability of a particular object belonging to a certain class (Y) given its observed feature values. 

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28Y%7Cfeatures%29%3D%5Cfrac%7B%5CPr%28Y%29%5CPr%28features%7CY%29%7D%7B%5CPr%28features%29%7D)

Training the bayesian model requires calculating Pr(features|Y<sub>i</sub>) for each label. For a dataset with a large number of features, this would be a time consuming calculation. However, under the assumption of independence, these class conditional probabilities can be estimated from the training data by calclulating the frequency of occurance of the feature f<sub>d</sub> in class Y<sub>i</sub> - 

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28features%7CY_i%29%20%3D%20%5CPr%28f_1%7CY_i%29*%20%5CPr%28f_2%7CY_i%29*%20.....%20%5CPr%28f_n%7CY_i%29%20%3D%20%5Cprod_%7Bd%3D1%7D%5E%7Bn%7D%20%5CPr%28f%7CY_i%29)

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28f_d%7CY_i%29%20%3D%20%5Cfrac%7B%5Ctext%7Bnumber%20of%20times%20feature%20%24f_d%24%20appears%20in%20class%20%24Y_i%24%20&plus;%20%24%5Calpha%24%7D%7D%7B%5Ctext%7Btotal%20count%20of%20all%20features%20in%20class%20%24Y_i%24%20&plus;%20%24%5Calpha%20N%24%7D%7D)

$\alpha$ is an additive smoothing parameter to compensate for edge cases (i.e. where Pr = 0) and N is the total number of features.

To decide between two labels/classes, e.g., Y<sub>1</sub>, Y<sub>2</sub>, all we need to do is caluculate the posterior probabilities for each class (denominator is the same for both and can be ignored) -  

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28Y_1%7Cfeatures%29%20%5Cpropto%20%5CPr%28Y_1%29%5CPr%28features%7CY_1%29%5C%5C%20%5CPr%28Y_2%7Cfeatures%29%20%5Cpropto%20%5CPr%28Y_2%29%5CPr%28features%7CY_2%29)

If Pr(Y<sub>1</sub>|features) > Pr(Y<sub>2</sub>|features) the sample is classified as Y<sub>1</sub> else Y<sub>2</sub>. If both probabilities are equal, then the model cannot asign a class and will require the assistance of the subject matter expert.



## Data Cleaning & Transformation

The data was cleaned and formatted prior to fitting the Naive Bayes model. A Bag of Words model was used to extract relevant features from the dataset. This method works by extracting all unique words from the text messages and storing them as vocabulary. A feature dataframe is formed by counting how many times the words in the vocabulary appear in a given text message. For e.g., let T1 & T2 be two text messages in the training dataset - 

```
T1: 'I love you all'
T2: 'Love conquers all'
```
Based on these texts, the vocabulary can be written as - 
```
V = {'I':1, 'love': 2, 'you':1, 'all': 2, 'conquers':1}
```

And the feature dataset formed is - 

|    | i | love | you | all | conquers |
|----|---|------|-----|-----|----------|
| T1 | 1 | 1    | 1   | 1   | 0        |
| T2 | 0 | 1    | 0   | 1   | 1        |

 
To do this, we removed all punctuations from the text messages and converted all words to lowercase. We then created a `vocabulary` list to store all non-redundant words in no specific order. There were 7,783 unique words (`n_vocabulary`) in our training data set. We used the list to transform the training data to the dataframe below - 


|      | Label |                                               SMS | barbie | bills | bthere | hiphop | absolutely | adrink | amused | ... | oble | genuine | establish | musicnews | mobilesdirect | salon | true18 | wn | fish | hit | ü |
|-----:|------:|--------------------------------------------------:|-------:|------:|-------:|-------:|-----------:|-------:|-------:|----:|-----:|--------:|----------:|----------:|--------------:|------:|-------:|---:|-----:|----:|---|
| 4363 |   ham |      [life, style, garments, account, no, please] |      0 |     0 |      0 |      0 |          0 |      0 |      0 |   0 |  ... |       0 |         0 |         0 |             0 |     0 |      0 |  0 |    0 |   0 | 0 |
| 2289 |   ham |                    [oh, did, you, charge, camera] |      0 |     0 |      0 |      0 |          0 |      0 |      0 |   0 |  ... |       0 |         0 |         0 |             0 |     0 |      0 |  0 |    0 |   0 | 0 |
| 1856 |   ham | [did, either, of, you, have, any, idea, s, do,... |      0 |     0 |      0 |      0 |          0 |      0 |      0 |   0 |  ... |       0 |         0 |         0 |             0 |     0 |      0 |  0 |    0 |   0 | 0 |


## Calculating Parameters

To be able to classify new messages, Naive Bayes theorem was used to answer the following probability equations - 

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28Spam%7Cw_1%2Cw_2%2C...w_n%29%20%5Cpropto%20%5CPr%28Spam%29%5Cprod_%7Bd%3D1%7D%5E%7Bn%7D%5CPr%28w_d%7CSpam%29%5C%5C%20%5CPr%28Ham%7Cw_1%2Cw_2%2C...w_n%29%20%5Cpropto%20%5CPr%28Ham%29%5Cprod_%7Bd%3D1%7D%5E%7Bn%7D%5CPr%28w_d%7CHam%29)

where - 

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28Spam%29%20%3D%20%5Cfrac%7B%5Ctext%7Blength%28spam%20messages%29%7D%7D%7B%5Ctext%7Blength%28training%20dataset%29%7D%7D%5C%5C%20%5CPr%28Ham%29%20%3D%20%5Cfrac%7B%5Ctext%7Blength%28ham%20messages%29%7D%7D%7B%5Ctext%7Blength%28training%20dataset%29%7D%7D%5C%5C)

and

![equation](https://latex.codecogs.com/gif.latex?%5CPr%28w_d%7CSpam%29%20%3D%20%5Cfrac%7BN_%7B%7Bw_d%7D%7CSpam%7D%20&plus;%20%5Calpha%7D%7BN_%7Bspam%7D%20&plus;%20%5Calpha%20N_%7Bvocabulary%7D%7D%5C%5C%20%5CPr%28w_d%7CHam%29%20%3D%20%5Cfrac%7BN_%7B%7Bw_d%7D%7CHam%7D%20&plus;%20%5Calpha%7D%7BN_%7Bham%7D%20&plus;%20%5Calpha%20N_%7Bvocabulary%7D%7D)

Here ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20N_%7B%7Bw_d%7D%7CSpam%7D) and ![equation](https://latex.codecogs.com/gif.latex?%5Cinline%20N_%7B%7Bw_d%7D%7CHam%7D) is the number of times the word w<sub>d</sub> appears in the spam or ham messages, N<sub>spam</sub> and N<sub>ham</sub> are the number of words in spam and ham messages, N<sub>vocabulary</sub> is the number of words in the vocabulary and $\alpha$ = 1 is the laplace smoothing parameter.

## Classify new messages

After caluclation all the probabilites, we wrote a function to classify new messages as spam or ham. The function - 
- Takes as input a new message.
- Calculates the probabilities Pr(spam|w_d) and Pr(ham|w_d) where w<sub>d</sub> are the words in the messages
- Classify the messages as -
    - Spam if Pr(spam|w_d) > Pr(ham|w_d)
    - Ham if Pr(spam|w_d) < Pr(ham|w_d)
- The function will request human help if Pr(spam|w_d) = Pr(ham|w_d)

### Accuracy

To calculate it's accuracy, we tested the model on the test dataset. The accuracy was calculated as -

![equation](https://latex.codecogs.com/gif.latex?Accuracy%20%3D%20%5Cfrac%7B%5Ctext%7Bnumber%20of%20correctly%20classified%20messages%7D%7D%7B%5Ctext%7Btotal%20number%20of%20messages%7D%7D)

The accuracy of our spam filter was 98.74%. Out of 1114 messages in the test dataset, the spam filter was able to correctly classify 1110 messgaes. This accuracy is much better than our goal of 80%. 

## Future analysis
In this project, we built a spam filter using Multinominal Naive Bayes algorithm. The filter had an accuracy of over 98%. Future analysis could include removing stop words (e.g. *and*, *a*, *the*, *so*). Stop words are words that are very common in text vocabulary but often don't add any meaning. 

## What I Learned

- What are conditional probabilities.
- Data extraction using Bag of Words model.
- Creating a spam filter using the Multinomial Naive Bayes algorithm.