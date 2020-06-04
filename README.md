# The *text2props* framework

This repository contains the source code of *text2props*, a framework to implement, train and evaluate models to 
estimate latent traits (e.g. difficulty) of questions from textual information.
The framework *text2props* was presented in the paper "*Introducing a framework to assess newly created questions with 
Natural Language Processing*" at the 21st International Conference on Artificial Intelligence in Education 
([AIED20](https://aied2020.nees.com.br/)).

If you use this framework, please cite the related paper (at the moment, only the pre-print version is available):
```
@article{benedetto2020introducing,
  title={Introducing a framework to assess newly created questions with Natural Language Processing},
  author={Benedetto, Luca and Cappelli, Andrea and Turrin, Roberto and Cremonesi, Paolo},
  journal={arXiv preprint arXiv:2004.13530},
  year={2020}
}
```

---

The repo is organized as follows:

- `scripts` contains some example scripts and the scripts used to obtain the results presented in the paper
- `tests` contains the unittests
- `text2props/data_validation` contains methods to validate the data (e.g., check whether a DF has the required columns)
- `text2props/evaluation` contains the code for evaluating the models.
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
