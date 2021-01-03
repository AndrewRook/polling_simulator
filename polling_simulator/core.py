import operator

from dataclasses import dataclass
from abc import ABC
from typing import Callable, Union, Dict

import numpy as np


class _Base(ABC):
    def __eq__(self, other: Union[int, float, str, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.eq)

    def __ne__(self, other: Union[int, float, str, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.ne)

    def __ge__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.ge)

    def __gt__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.gt)

    def __le__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.le)

    def __lt__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.lt)

    def __and__(self, other: Union["Variable", "Segmentation"]):
        return Segmentation(self, other, operator.and_)

    def __or__(self, other: Union["Variable", "Segmentation"]):
        return Segmentation(self, other, operator.or_)


class Variable(_Base):
    """
    A named variable, which can be used in Segmentations.

    Parameters
    ----------
    name: The name of the variable.
    data_generator: A function which takes in a single integer, the number of data points
    to generate, and returns that many data points with appropriately randomly generated
    values. This allows for arbitrary generation strategies, from simply coin flips to complex
    distributions.
    """
    def __init__(
        self,
        name: str,
        data_generator: Callable[[int], Union[np.int64, np.float64, np.object_]]
    ):
        self.name = name
        self.data_generator = data_generator

    def __str__(self):
        return self.name


class Segmentation(_Base):
    """
    An object that represents a comparison between any of a Variable, a constant (e.g. int or string),
    and/or another Segmentation.

    Parameters
    ----------
    left: The left side of the comparison (i.e. the "age" of "age > 21")
    right: The right side of the comparison (i.e. the "21" of "age > 21")
    comparator: The comparison (i.e. the ">" of "age > 21"). Can be one of:
        >, >=, <, <=, ==, !=, &, |
    """
    def __init__(
            self,
            left: Union[int, float, str, Variable, "Segmentation"],
            right: Union[int, float, str, Variable, "Segmentation"],
            comparator
    ):
        self.left = left
        self.right = right
        self.comparator = comparator

    def segment(self, df):
        """
        Perform the segmentation on a pandas DataFrame.

        Parameters
        ----------
        df: A pandas DataFrame, which at least has columns with names corresponding to
        all Variables used in the Segmentation (and any sub-segmentations).

        Returns
        -------
        A numpy array of booleans, where ``True`` indicates the row of the input ``df`` is
        in the segmentation and ``False`` means it is not.
        """
        # If Segmentation, first evaluate Segmentation
        left = self.left if issubclass(self.left.__class__, Segmentation) is False else self.left.segment(df)
        right = self.right if issubclass(self.right.__class__, Segmentation) is False else self.right.segment(df)
        # If Variable, replace with dataframe column
        left = left if issubclass(left.__class__, Variable) is False else df[left.name]
        right = right if issubclass(right.__class__, Variable) is False else df[right.name]

        return self.comparator(left, right)

    @property
    def variables(self):
        """
        Parse the Segmentation, recursively if necesary, to obtain a list of
        all Variables used in it. This is useful for things like building a
        DataFrame containing these variables, or validating that a DataFrame has the
        right variables in it.

        Returns
        -------
        A list of all unique Variable instances used in the Segmentation.
        """
        all_variables = []
        if issubclass(self.left.__class__, Variable):
            all_variables.append(self.left)
        elif issubclass(self.left.__class__, Segmentation):
            all_variables += self.left.variables

        if issubclass(self.right.__class__, Variable):
            all_variables.append(self.right)
        elif issubclass(self.right.__class__, Segmentation):
            all_variables += self.right.variables

        unique_variables = _uniquefy_variables(all_variables)
        return unique_variables

    def __str__(self):
        comparator_map = {
            operator.gt: ">",
            operator.ge: ">=",
            operator.lt: "<",
            operator.le: "<=",
            operator.eq: "==",
            operator.ne: "!=",
            operator.and_: "&",
            operator.or_: "|"
        }
        left = f"({self.left})" if issubclass(self.left.__class__, Segmentation) else f"{self.left}"
        right = f"({self.right})" if issubclass(self.right.__class__, Segmentation) else f"{self.right}"
        return f"{left} {comparator_map[self.comparator]} {right}"


def _uniquefy_variables(non_unique_variables):
    # Have to use this crazy explicit nested loop because Variable
    # overrides the __eq__ method
    unique_variables = []
    for variable in non_unique_variables:
        already_used = False
        for unique_variable in unique_variables:
            if variable is unique_variable:
                already_used = True
                break
        if not already_used:
            unique_variables.append(variable)

    return unique_variables


@dataclass
class Demographic:
    """
    A simple class that contains all the necessary information about a demographic

    Parameters
    ----------
    turnout_likelihood: a number between 0 and 1 that corresponds to the fractional
        probability that someone in this demographic will vote.
    response_likelihood: a number between 0 and 1 that corresponds to the fractional
        probability that someone in this demographic will respond when contacted by
        a pollster. (Note: If you set up a poll with multiple contact attempts, this
        likelihood governs each attempt; if you have multiple contact attempts you
        will have a higher chance of a response than would be indicated by this variable.
    candidate_preference: A mapping between candidate identifier and the fraction of people
        in the demographic who would vote for them. For example, if a Democrat has 60% of the
        support in this demographic compared to 40% for a Republican, you'd enter
        ``{"Democrat": 0.6, "Republican": 0.4}``
    population_segmentation: The Segmentation needed to identify who is in this demographic.
    """
    turnout_likelihood: float
    response_likelihood: float
    candidate_preference: Dict[str, float] # TODO: ensure these sum to 1
    population_segmentation: Segmentation

    def get_population_in_demographic(self, population):
        """
        A small helper function to identify how prevalent the demographic is in a population.

        Parameters
        ----------
        population: A DataFrame containing all individuals in the population.

        Returns
        -------
        The number of individuals in that population which are in this demographic.
        """
        in_demographic = self.population_segmentation.segment(population)
        return np.sum(in_demographic)
