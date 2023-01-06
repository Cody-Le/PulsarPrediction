# Pulsar Candidate Classification Model

This paper was done as a research for the IB as a Math Approaches and Analysis Internal Assessment. 

![Pulsar Star Diagram](https://github.com/Cody-Le/PulsarPrediction/blob/main/ns_pulsar_diagram.png?raw=true)

The aim for the research was to classify Pulsar Star Candidate using a logistic regression model. The dataset used in the research was the [HTRU2 Dataset](https://archive.ics.uci.edu/ml/datasets/HTRU2). You can view the paper [here](https://github.com/Cody-Le/PulsarPrediction/blob/main/Math%20IApdf.pdf).


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
