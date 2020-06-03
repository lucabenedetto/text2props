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

- `text2props/data_validation` contains the methods for validate and check the data (e.g., check whether a 
DataFrame has the required columns)
- `text2props/evaluation` contains the code for evalutating the models.
- `text2props/model` contains the definition of the classes used to implement the models.
- `text2props/modules` contains the definition of the classes used to implement the modules and their components.
- `text2props/utils` contains some utility code used in other parts of the package.

---

## How to use

First of all, you have to install the package.
If you want to do so in a new virtual environment, you can use the following commands:

```
conda create -n venv-text2props python=3.7 pip
pip setup.py install
```

Then, you can run the example scripts (no dataset is provided within this repo, though, you have to get your own):

```
python scripts/example_script.py
```

---

## Data format

Some datasets (not present in this repository) are required to run the scripts.
The required format is as follows, using the same names as in the `constants.py` file.

* **answers dataframe**: contains the list of interactions between students and questions (i.e. the list of answers
given by students to the questions). It has the following columns: `[S_ID, TIMESTAMP, CORRECT, Q_ID]`, which are the
student ID, the timestamp of the interaction, the correctness of the answer, and the question ID.

* **questions dataframe**: contains the textual information for all the questions. It has the following columns:
`[Q_ID, Q_TEXT, CORRECT_TEXTS, WRONG_TEXTS]`, which are the question ID, the text of the question, the list of the text
of the correct choices (possibly 1 element long), the list of the texts of the wrong choices

* **questions' latent traits dataframe**: contains the latent traits (if already known) of the questions. It has the
following columns: `[Q_ID, LATENT_TRAIT]`, which are the question ID and the value of the latent traits.


---

## Tests

To launch tests:

`python -m unittest discover tests -v`
