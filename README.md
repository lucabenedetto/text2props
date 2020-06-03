# The *text2props* framework

This repository contains the source code of text2props, a framework to estimate latent traits of questions from textual information.
text2props was presented in the paper "Introducing a framework to assess newly created questions with Natural Language Processing" at the 2020 International conference of Artificial Intelligence in Education (AIED20).

`text2props` framework, which can be used to implement, train and evaluate models 
for estimating questions' properties (i.e. latent traits) from textual information.
In particular, the models implemented with this framework estimate questions' latent traits using the text of the 
questions and, in the case of Multiple Choice Questions (MCQ), the text of the possible choices.
Using `text2props`, one or more latent traits can be estimated at the same time.

The model implemented with `text2props` have two components:

1. latent traits calibrator - performs question calibration, by estimating ground truth latent traits, which are then 
used to train and evaluate the estimators that estimate latent traits from textual information
2. estimator of latent traits form text - performs the actual estimation of the latent traits from text

---

This repo is organized as follows:

- `text2props/data_validation` contains the methods for validate and check the data (for instance, check whether a 
DataFrame has the required columns)
- `text2props/evaluation` contains the code for evalutating the models.
- `text2props/model` contains the definition of the classes used to implement the models.
- `text2props/modules` contains the definition of the classes used to implement the modules and their components.
- `text2props/utils` contains some utility code used in other parts of the package.

---