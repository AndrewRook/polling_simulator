import warnings
from typing import Iterable, Callable, List

import numpy as np
import pandas as pd

from polling_simulator.core import Demographic, Variable, _uniquefy_variables


def generate_electorate(num_people: int, demographics: Iterable[Demographic]):
    variables_used = []
    candidates = set()
    for demographic in demographics:
        variables_used += demographic.population_segmentation.variables
        candidates.update(list(demographic.candidate_preference.keys()))
    candidates = list(candidates)  # converting to list allows for easier indexing later

    variables_used = _uniquefy_variables(variables_used)
    # Create the basic values of the electorate
    electorate = pd.DataFrame({
        variable.name: variable.data_generator(num_people)
        for variable in variables_used
    })
    # Add in columns for turnout, response, and candidate, based on demographics
    electorate = pd.concat(
        [electorate, generate_demographic_features_of_population(electorate, demographics, candidates)],
        axis=1
    )

    return electorate


def generate_demographic_features_of_population(
        population: pd.DataFrame, demographics: Iterable[Demographic], candidates: List[str]
):
    # Start with dummy values for features
    demographic_features = pd.DataFrame({
        "turnout_likelihood": np.ones(len(population), dtype=np.float) * -1,
        "response_likelihood": np.ones(len(population), dtype=np.float) * -1,
        "candidate_preference": pd.Categorical([candidates[0]] * len(population), categories=candidates)
    })
    population_already_in_demographic = np.zeros(len(population), dtype=np.bool_)
    for demographic in demographics:
        population_in_demographic = demographic.population_segmentation.segment(population)
        if np.sum(population_in_demographic & population_already_in_demographic) != 0:
            # If someone is in multiple demographics, bail out
            raise ValueError(
                f"""
Some demographics overlap. Examples include:
{population[population_already_in_demographic & population_in_demographic]}
                """
                )
        demographic_features.loc[population_in_demographic, "turnout_likelihood"] = demographic.turnout_likelihood
        demographic_features.loc[population_in_demographic, "response_likelihood"] = demographic.response_likelihood
        demographic_features.loc[population_in_demographic, "candidate_preference"] = np.random.choice(
            np.array(list(demographic.candidate_preference.keys())),
            np.sum(population_in_demographic),
            replace=True,
            p=np.array(list(demographic.candidate_preference.values()))
        )
        population_already_in_demographic = population_in_demographic | population_already_in_demographic

    if np.sum(population_already_in_demographic) != len(population_already_in_demographic):
        # If someone is in NO demographics, bail out
        raise ValueError(f"""
Demographics do not cover entire population. Examples include:
{population[population_already_in_demographic == False].head()}
        """)

    return demographic_features


def run_election(population: pd.DataFrame):
    does_person_vote = population["turnout_likelihood"] > np.random.random(len(population))
    votes = population.loc[does_person_vote, "candidate_preference"].value_counts()

    return votes


def run_elections(num_elections: int, population: pd.DataFrame):
    election_results = pd.concat([
        run_election(population).to_frame().T
        for _ in range(num_elections)
    ])
    # Handle the edge case where sometimes a candidate gets zero votes
    election_results = election_results.fillna(0).astype(np.int)
    # Remove categorical column indexer as it's not needed (and makes life harder)
    election_results.columns = election_results.columns.astype("str")
    return election_results.reset_index(drop=True)


def run_poll(
        num_to_poll: int,
        electorate: pd.DataFrame,
        assumed_demographics: Iterable[Demographic],
        sampling_strategy: Callable, aggregation_strategy: Callable):
    shuffled_electorate = electorate.sample(frac=1).reset_index(drop=True)
    poll_responders, poll_nonresponders = sampling_strategy(num_to_poll, shuffled_electorate)

    poll_results = aggregation_strategy(poll_responders, poll_nonresponders)
    poll_percentages = poll_results / poll_results.sum()
    return poll_percentages


def run_polls(
        num_polls: int,
        num_to_poll: int,
        electorate: pd.DataFrame,
        assumed_demographics: Iterable[Demographic],
        sampling_strategy: Callable,
        aggregation_strategy: Callable):
    poll_results = [
        run_poll(
            num_to_poll,
            electorate,
            assumed_demographics,
            sampling_strategy,
            aggregation_strategy
        )
        for _ in range(num_polls)
    ]
    return pd.concat(poll_results, axis=1).T.reset_index(drop=True)