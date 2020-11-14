import numpy as np
import operator
import pandas as pd

from abc import ABC
from dataclasses import dataclass
from functools import partial
from scipy import stats
from typing import Callable, Iterable, Tuple, Union


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


@dataclass(frozen=True)
class Demographic:
    turnout_likelihood: float
    response_likelihood: float
    candidate_preference: float
    population_segmentation: Segmentation


def generate_electorate(num_people: int, demographics: Iterable[Tuple[Demographic, float]]):
    total_percentage_of_electorate = 0
    variables_used = []
    for demographic in demographics:
        if len(demographic) != 2:
            raise ValueError("demographics must be an iterable of (Demographic, electorate %)")
        if demographic[1] > 1:
            raise ValueError("demographic percentages must be <= 1")

        total_percentage_of_electorate += demographic[1]
        variables_used += demographic[0].population_segmentation.variables

    if abs(1 - total_percentage_of_electorate) > 1e-6:
        raise ValueError(f"total electorate % must equal 1, not {total_percentage_of_electorate}")

    variables_used = _uniquefy_variables(variables_used)
    electorate = pd.concat([
        _generate_demographic(round(num_people * percent_in_demographic), demographic, variables_used)
        for demographic, percent_in_demographic in demographics
    ])


def _generate_demographic(num_people: int, demographic: Demographic, variables: Iterable[Variable]):
    initial_demographic = pd.DataFrame({
        variable.name: variable.data_generator(num_people)
        for variable in variables
    })
    breakpoint()


def convert_generic_scipy_distribution(distribution, *args, **kwargs):
    distro = distribution(*args, **kwargs)
    return distro.rvs


def truncated_gaussian_distribution(mean, sigma, lower_clip, upper_clip):
    a = (lower_clip - mean) / sigma
    b = (upper_clip - mean) / sigma
    return convert_generic_scipy_distribution(stats.truncnorm, a, b, loc=mean, scale=sigma)


if __name__ == "__main__":
    age = Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))
    gender = Variable("gender", partial(np.random.choice, ["M", "F"], replace=True, p=[0.49, 0.51]))
    young_men = Demographic(
        0.5,
        0.1,
        0,
        (age < 40) & (gender == "M")
    )
    old_men = Demographic(
        0.7,
        0.2,
        1,
        (age >= 40) & (gender == "M")
    )
    young_women = Demographic(
        0.6,
        0.05,
        0,
        (age < 40) & (gender == "F")
    )
    old_women = Demographic(
        0.8,
        0.2,
        0,
        (age >= 40) & (gender == "F")
    )
    generate_electorate(
        1000,
        [
            (young_men, 0.25),
            (old_men, 0.25),
            (young_women, 0.25),
            (old_women, 0.25)
        ]
    )

# @dataclass
# class Electorate:
#     demographics: Iterable[Demographic]
#
#     def generate_electorate