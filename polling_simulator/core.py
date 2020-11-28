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
    def __init__(
        self,
        name: str,
        data_generator: Callable[[int], Union[np.int64, np.float64, np.object_]]
    ):
        self.name = name
        self.data_generator = data_generator


class Segmentation(_Base):
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
        # If Segmentation, first evaluate Segmentation
        left = self.left if issubclass(self.left.__class__, Segmentation) is False else self.left.segment(df)
        right = self.right if issubclass(self.right.__class__, Segmentation) is False else self.right.segment(df)
        # If Variable, replace with dataframe column
        left = left if issubclass(left.__class__, Variable) is False else df[left.name]
        right = right if issubclass(right.__class__, Variable) is False else df[right.name]

        return self.comparator(left, right)

    @property
    def variables(self):
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
    population_percentage: float
    turnout_likelihood: float
    response_likelihood: float
    candidate_preference: Dict[str, float] # TODO: ensure these sum to 1
    population_segmentation: Segmentation
