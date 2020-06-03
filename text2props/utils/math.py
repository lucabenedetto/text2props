import numpy as np
from ..constants import DISCRIMINATION_COEFFICIENT


def item_response_function(difficulty, skill, discrimination, guess, slip) -> float:
    """
    Computes the logistic function for the given arguments and returns a float. The initial np.product is necessary for
    the multidimensional case.
    :param difficulty:
    :param skill:
    :param discrimination:
    :param guess:
    :param slip:
    :return:
    """
    return np.product(
        np.add(
            guess,
            np.divide(
                1.0 - np.add(guess, slip),
                1.0 + np.exp(-1.7 * np.multiply(discrimination, np.subtract(skill, difficulty)))
            )
        )
    )


def inverse_item_response_function(difficulty, skill, discrimination, guess, slip) -> float:
    """
    Computes 1 - logistic function for the given arguments and returns a float.
    :param difficulty: difficulty of the item
    :param skill: the skill of the user
    :param discrimination: the discrimination factor
    :param guess: the guess factor
    :param slip: the slip factor
    :return:
    """
    return 1.0 - item_response_function(difficulty, skill, discrimination, guess, slip)


def information_function(b, theta, discrimination, guess=0, slip=0) -> float:
    """
    Information function of a question: I(theta) = (P'(theta))**2/(P(theta)*Q(theta)), where Q(theta) = 1 - P(theta)
    :param b: difficulty of the item
    :param theta: the skill of the user
    :param discrimination: the discrimination factor, mono-dimensional
    :param guess: the guess factor, mono-dimensional
    :param slip: the slip factor, mono-dimensional
    :return: result of the Information Function
    """
    return np.divide(
        np.square(derivative_item_response_function(b, theta, discrimination, guess, slip)),
        (
                item_response_function(b, theta, discrimination, guess, slip)
                * inverse_item_response_function(b, theta, discrimination, guess, slip)
        )
    )


def derivative_item_response_function(b, theta, discrimination, guess, slip) -> float:
    """
    Computes the derivative of the item_response_function.
    :param b: difficulty of the item
    :param theta: the skill of the user
    :param discrimination: the discrimination factor, mono-dimensional
    :param guess: the guess factor, mono-dimensional
    :param slip: the slip factor, mono-dimensional
    :return: the result of the derivative of the item response function
    """
    x = np.exp(-DISCRIMINATION_COEFFICIENT * discrimination * (theta[0]-b[0]))
    return np.divide((1.-guess-slip) * x * (-DISCRIMINATION_COEFFICIENT) * discrimination, np.square(1 + x**2))
