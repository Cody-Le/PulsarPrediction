<p align = "center">
# Pulsar Candidate Classification Model

This paper was done as a research for the IB Math Approaches and Analysis Internal Assessment. 

<img width="200" src="https://github.com/Cody-Le/PulsarPrediction/blob/main/ns_pulsar_diagram.png?raw=true" alt="Pulsar Star Diagram">
</p>

The aim for the research was to classify Pulsar Star Candidate using a logistic regression model. The dataset used in the research was the [HTRU2 Dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2). You can view the paper [here](https://github.com/Cody-Le/PulsarPrediction/blob/main/Math%20IApdf.pdf).


## Requirements

To install requirements:


The 3 libraries the model use is pandas, matplotlib and numpy. You can decide to install as a package by installing conda or to install them using pip as follow:

```setup
pip install numpy pandas matplotlib
```
The data can be obtained by downloading the data from the UCI Archive or you can use the data already contained in the repository. 


## Training and Evaluating

To train the model(s) in the paper, run this command:

```Run the program
python LogisticRegression.py
```
Running this program should train and return the evaluation for the model. Keep in mind, the data will need to feature scaled first. Other hyperparameter such as starting weights or amount of candidates is used for training and testing can be change in the class object and the running loop respectively.  

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
