import warnings
from typing import Iterable, Callable, List

import numpy as np
import pandas as pd

from polling_simulator.core import Demographic, Variable, _uniquefy_variables


def generate_electorate(num_people: int, demographics: Iterable[Demographic]):
    """
    Construct an electorate based on a set of demographics. In addition to setting values based
    on which demographic each person belongs to (turnout likelihood, candidate preference, etc),
    for each person in the electorate,
    randomly determine values for all ``Variable``s present in any demographics based on their
    data generator functions.

    Parameters
    ----------
    num_people: The number of people you want to be in the electorate.
    demographics: The demographics you want to use to generate the population.

    Returns
    -------
    A pandas DataFrame, where each row is a person in the electorate and the columns contain information
    about their demographics and their voting preferences.
    """
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
    """
    A helper function, used in ``generate_electorate``, to fill in columns based on demographic
    information. Should not need to be used independently.
    """
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
    """
    Run an election, based on a simulated population.

    Parameters
    ----------
    population: A dataframe, likely created by ``generate_electorate``, which contains
    (at minimum) turnout likelihoods and candidate preferences.

    Returns
    -------
    A pandas DataFrame with the number of votes for each candidate.
    """
    does_person_vote = population["turnout_likelihood"] > np.random.random(len(population))
    votes = population.loc[does_person_vote, "candidate_preference"].value_counts()

    return votes


def run_elections(num_elections: int, population: pd.DataFrame):
    """
    Run multiple elections, using the same population. Good for understanding what
    the distribution of possible election outcomes can be.

    Parameters
    ----------
    num_elections: Number of elections to simulate.
    population: A dataframe, likely created by ``generate_electorate``, which contains
    (at minimum) turnout likelihoods and candidate preferences.

    Returns
    -------
    A pandas DataFrame, where each row represents an election and the columns show the
    votes received for each candidate.
    """
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
        sampling_strategy: Callable, aggregation_strategy: Callable):
    """
    Run a single poll based on an electorate, a sampling strategy, and an aggregation strategy.

    Parameters
    ----------
    num_to_poll: Number of people to poll. Note that the actual number of people polled will
        depend on which ``sampling_strategy`` is chosen.
    electorate: A dataframe, likely created by ``generate_electorate``, which contains
        all relevant information about individual voters in the electorate you want to poll.
    sampling_strategy: A function to determine who to poll. It must take two arguments: The number of people to
        poll and a shuffled electorate dataframe, and return two DataFrames containing certain rows of
        the input electorate DataFrame — one for poll responders and one for anyone who was contacted but did not
        respond. Usually the output of one of the functions in the ``sampling`` module.
    aggregation_strategy: A function that determines how to aggregate poll responses. It must take
        two arguments: A DataFrame of information about poll responders and a DataFrame of
        information about poll non-responders, and return a DataFrame where the rows are the candidates
        and the column shows the number of responders who supported that candidate. Usually the output of
        one of the functions in the ``aggregation`` module.

    Returns
    -------
    A pandas DataFrmae showing the fraction of support for each candidate.
    """
    shuffled_electorate = electorate.sample(frac=1).reset_index(drop=True)
    poll_responders, poll_nonresponders = sampling_strategy(num_to_poll, shuffled_electorate)

    poll_results = aggregation_strategy(poll_responders, poll_nonresponders)
    poll_percentages = poll_results / poll_results.sum()
    return poll_percentages


def run_polls(
        num_polls: int,
        num_to_poll: int,
        electorate: pd.DataFrame,
        sampling_strategy: Callable,
        aggregation_strategy: Callable):
    """
    Run multiple polls on the same electorate, with the same sampling and aggregation strategies
    Parameters
    ----------
    num_polls: Number of polls to run.
    num_to_poll: Number of people to poll. Note that the actual number of people polled will
        depend on which ``sampling_strategy`` is chosen.
    electorate: A dataframe, likely created by ``generate_electorate``, which contains
        all relevant information about individual voters in the electorate you want to poll.
    sampling_strategy: A function to determine who to poll. It must take two arguments: The number of people to
        poll and a shuffled electorate dataframe, and return two DataFrames containing certain rows of
        the input electorate DataFrame — one for poll responders and one for anyone who was contacted but did not
        respond. Usually the output of one of the functions in the ``sampling`` module.
    aggregation_strategy: A function that determines how to aggregate poll responses. It must take
        two arguments: A DataFrame of information about poll responders and a DataFrame of
        information about poll non-responders, and return a DataFrame where the rows are the candidates
        and the column shows the number of responders who supported that candidate. Usually the output of
        one of the functions in the ``aggregation`` module.

    Returns
    -------
    A pandas DataFrame, where each row is a poll and each column corresponds to the fraction of particpants
    who supported that candidate.
    """
    poll_results = [
        run_poll(
            num_to_poll,
            electorate,
            sampling_strategy,
            aggregation_strategy
        )
        for _ in range(num_polls)
    ]
    return pd.concat(poll_results, axis=1).T.reset_index(drop=True)
