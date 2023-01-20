<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=1&height=200&section=header&text=Normalizing%20Flow%20and%20Class%20Imbalance&fontSize=40&animation=fadeIn&fontColor=000000" />

# A novel machine learning approach using normalizing flow to address extreme class imbalance problems in personal health records
Yeongmin Kim\*, Wongyung Choi\*, Woojeong Choi\*, Grace Ko\*, Younghee Lee

## Abstract
The class imbalance problem in Personal Health Records (PHRs) limits performance when analyzing such data using machine learning algorithms. To tackle this challenge, we developed a conditional normalizing flow algorithm with semi-supervised anomaly detection. We compared our novel approach with Light Gradient Boosting Machine, which is widely used in classification tasks, and also with a one-class suppot vector machine, Gaussian mixture model, and autoencoder, which are representative semi-supervised anomaly detection algorithms. To train and evaluate models, we used PHR data comprised of medical check-up, life-log, and genetic testing data, and defined as the target variables six chronic diseases (e.g., obesity, diabetes). We found the proposed normalizing flow model to outperform the other tested models when used on an extremely imbalanced class. We further demonstrated that classification models with class weights performed poorly in scenarios with low base rate regardless of a biological context, unlike the proposed normalizing flow model.

## Datasets
The data underlying this article cannot be shared publicly due to the privacy of individuals that participated in the study.

## Code Organization
constant.py: Usuful constants including key hyperparameters
main.py: Execution of an overall process
model_flow.py: Implementation of the architecture of normalizing flow
train.py: Data preparation and train of the model
utils.py: Defines CustomCosineAnnealingWarmUpRestarts.

## Installation

## How to run
--seed: Fixs seed as the number (Default: 0)
--use-altub: Operates [AltUB](https://arxiv.org/abs/2210.14913). An option depending on properties of a dataset.
--eval-all: Evaluates for every possible test set. Being highly time-consuming, not recommended generally.
--category: A type of disease
'''
python3 main.py --data data/phr_data.csv --seed 0 --use-altub --eval-all --category diabetes
'''

## Architecture of Normalizing Flow Model on PHR
![PHR-NF](./images/fig-flow-flow.svg)
