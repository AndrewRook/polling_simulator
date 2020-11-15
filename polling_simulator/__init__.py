import numpy as np
import operator
import pandas as pd
import warnings

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from scipy import stats
from typing import Callable, Dict, Iterable, Tuple, Union


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


def generate_electorate(num_people: int, demographics: Iterable[Demographic]):
    total_percentage_of_electorate = 0
    variables_used = []
    for demographic in demographics:
        if demographic.population_percentage > 1: # should be part of Demographic.__init__
            raise ValueError("demographic percentages must be <= 1")

        total_percentage_of_electorate += demographic.population_percentage
        variables_used += demographic.population_segmentation.variables

    if abs(1 - total_percentage_of_electorate) > 1e-6:
        raise ValueError(f"total electorate % must equal 1, not {total_percentage_of_electorate}")

    variables_used = _uniquefy_variables(variables_used)
    electorate = pd.concat(
        [
            _generate_demographic_population(
                round(num_people * demographic.population_percentage), demographic, variables_used
            )
            for demographic in demographics
        ], ignore_index=True
    )

    return electorate


def _generate_demographic_population(num_people: int, demographic: Demographic, variables: Iterable[Variable]):
    # Start with initial guess based on desired population, which will almost certainly be too small
    num_people_to_generate = num_people * 2
    initial_population = pd.DataFrame({
        variable.name: variable.data_generator(num_people_to_generate)
        for variable in variables
    })
    initial_segmentation_map = demographic.population_segmentation.segment(initial_population)
    demographic_population = initial_population[initial_segmentation_map]

    # Figure out how bad that guess was
    accepted_fraction = len(demographic_population) / num_people_to_generate
    if accepted_fraction < 0.1:
        warnings.warn(f"demographic is rare enough that {accepted_fraction:.2%} of random data is rejected")
    if len(demographic_population) == 0:
        raise ValueError("demographic does not appear to exist in population.")

    # Adapt guess size and then do a dumb loop to fill out the dataframe
    num_people_to_generate = round(num_people_to_generate / accepted_fraction)
    while len(demographic_population) < num_people:
        test_population = pd.DataFrame({
            variable.name: variable.data_generator(num_people_to_generate)
            for variable in variables
        })
        segmentation_map = demographic.population_segmentation.segment(test_population)
        additional_demographic_population = test_population[segmentation_map]
        demographic_population = pd.concat(
            [demographic_population, additional_demographic_population], ignore_index=True
        )

    # Making sure to get exactly the right number:
    demographic_population = demographic_population.head(num_people)

    # Adding additional necessary columns:
    demographic_population["turnout_likelihood"] = demographic.turnout_likelihood
    demographic_population["response_likelihood"] = demographic.response_likelihood
    demographic_population["candidate_preference"] = np.random.choice(
        np.array(list(demographic.candidate_preference.keys())),
        len(demographic_population),
        replace=True,
        p=np.array(list(demographic.candidate_preference.values()))
    )
    return demographic_population


def convert_generic_scipy_distribution(distribution, *args, **kwargs):
    distro = distribution(*args, **kwargs)
    return distro.rvs


def truncated_gaussian_distribution(mean, sigma, lower_clip, upper_clip):
    a = (lower_clip - mean) / sigma
    b = (upper_clip - mean) / sigma
    return convert_generic_scipy_distribution(stats.truncnorm, a, b, loc=mean, scale=sigma)


def run_election(population: pd.DataFrame):
    does_person_vote = population["turnout_likelihood"] > np.random.random(len(population))
    votes = population.loc[does_person_vote, "candidate_preference"].value_counts()

    return votes


def run_multiple_elections(num_elections: int, population: pd.DataFrame):
    election_results = pd.concat([
        run_election(population).to_frame().T
        for _ in range(num_elections)
    ])
    election_results = election_results.fillna(0).astype(np.int)
    return election_results.reset_index(drop=True)


def run_poll(
        num_to_poll: int,
        electorate: pd.DataFrame,
        assumed_demographics: Iterable[Demographic],
        polling_strategy, sampling_strategy):
    shuffled_electorate = electorate.sample(frac=1).reset_index(drop=True)
    does_respond = shuffled_electorate["response_likelihood"] > np.random.random(len(shuffled_electorate))
    poll_responders = shuffled_electorate[does_respond].head(num_to_poll)


if __name__ == "__main__":
    age = Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))
    gender = Variable("gender", partial(
        np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.49, 0.51])
    ))
    young_men = Demographic(
        0.25,
        0.5,
        0.1,
        {"a": 0.99, "c": 0.01},
        (age < 40) & (gender == "M")
    )
    old_men = Demographic(
        0.25,
        0.7,
        0.2,
        {"b": 1},
        (age >= 40) & (gender == "M")
    )
    young_women = Demographic(
        0.25,
        0.6,
        0.05,
        {"a": 1},
        (age < 40) & (gender == "F")
    )
    old_women = Demographic(
        0.25,
        0.8,
        0.2,
        {"a": 1},
        (age >= 40) & (gender == "F")
    )
    np.random.seed(123)
    electorate = generate_electorate(
        1000,
        [
            young_men, old_men, young_women, old_women
            # (young_men, 0.25),
            # (old_men, 0.25),
            # (young_women, 0.25),
            # (old_women, 0.25)
        ]
    )
    results = run_multiple_elections(10, electorate)
    breakpoint()

# @dataclass
# class Electorate:
#     demographics: Iterable[Demographic]
#
#     def generate_electorate