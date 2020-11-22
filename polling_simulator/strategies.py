import numpy as np
import pandas as pd

from typing import Callable, Union, Tuple

def post_stratified_sample_predefined():
    """ie "I'm going to call 1000 people at random, and then we'll weight the responses
    we get by assumed demographic representation
    """


def post_stratified_sample_guaranteed():
    """ie "I'm going to call people at random until I get 1000 responses, and then we'll
    weight the responses we get by assumed demographic representation
    """


def predefined_sample(n_sample, shuffled_electorate, max_num_attempts):
    """ie I'm going to call 1000 people at random, and just use those who answer"""
    if n_sample > len(shuffled_electorate):
        raise ValueError(f"number of samples ({n_sample}) greater than electorate ({len(shuffled_electorate)})")
    people_called = shuffled_electorate.head(n_sample)
    does_respond = people_called["response_likelihood"] > np.random.random(len(people_called))
    poll_responders = people_called[does_respond].reset_index(drop=True)
    poll_non_responders = people_called[does_respond == False].reset_index(drop=True)
    return poll_responders, poll_non_responders


def guaranteed_sample(n_sample, shuffled_electorate, max_num_attempts):
    """ie I'm going to call people at random until I get 1000 responses"""
    does_respond, num_attempts_required = _get_responses(
        shuffled_electorate["response_likelihood"], max_num_attempts
    )
    # TODO: figure out how to add the info about number of attempts required
    cumulative_responses = np.cumsum(does_respond)
    if cumulative_responses[-1] < n_sample:
        raise ValueError(
            f"number of samples ({n_sample}) greater than number of poll responders ({cumulative_responses[-1]})"
        )
    people_contacted = shuffled_electorate[cumulative_responses <= n_sample]
    does_respond = does_respond[cumulative_responses <= n_sample]
    num_attempts_required = num_attempts_required[cumulative_responses <= n_sample]
    poll_responders = people_contacted.loc[does_respond, :].copy(deep=True)
    poll_responders["num_contact_attempts"] = num_attempts_required[does_respond]
    poll_nonresponders = people_contacted.loc[does_respond == False, :].copy(deep=True)
    poll_nonresponders["num_contact_attempts"] = max_num_attempts

    return poll_responders, poll_nonresponders


def _get_responses(
        response_likelihoods: Union[pd.Series, np.float64], max_num_attempts: int
):
    # -> Tuple[np.ndarray[np.bool_], np.ndarray[np.int64]]: # Not working, not sure why
    """Figure out whether or not someone responds after N attempts. Return both whether or
    not they responded but also the number of attempts made."""
    if max_num_attempts < 1:
        raise ValueError("max_num_attempts must be a positive integer")
    realized_response_matrix = np.random.random((max_num_attempts, len(response_likelihoods)))
    if isinstance(response_likelihoods, pd.Series):
        response_likelihoods = response_likelihoods.values

    did_respond_matrix = realized_response_matrix < response_likelihoods
    did_respond = did_respond_matrix.any(axis=0)
    num_attempts_required = did_respond_matrix.argmax(axis=0) + 1
    num_attempts_required[did_respond == False] = -1
    return did_respond, num_attempts_required


def pre_stratified_sample(
        n_sample: int,
        shuffled_electorate: pd.DataFrame,
        max_num_attempts: int,
        assumed_demographics,
        sampling_strategy: Callable
):
    n_sample_per_demographic = [
        round(n_sample * demographic.population_percentage)
        for demographic in assumed_demographics
    ]
    population_per_demographic = [
        shuffled_electorate.loc[
            demographic.population_segmentation.segment(shuffled_electorate), :
        ].copy(deep=True)
        for demographic in assumed_demographics
    ]
    poll_responders = [
        sampling_strategy(sample_size, demo_population, max_num_attempts)
        for sample_size, demo_population in zip(n_sample_per_demographic, population_per_demographic)
    ]
    return pd.concat(poll_responders).reset_index(drop=True)

#
# def predefined_sample_pre_stratified(n_sample, shuffled_electorate, demographics, max_num_attempts):
#     """ie I'm going to call 1000 people, randomly selected based on assumed demographic representation"""
#     n_sample_per_demo = [
#         round(n_sample * demographic.population_percentage)
#         for demographic in demographics
#     ]
#     population_per_demo = [
#         shuffled_electorate.loc[demographic.population_segmentation.segment(shuffled_electorate), :]
#         for demographic in demographics
#     ]
#     poll_responders = [
#         predefined_sample(sample_size, demo_population, None)
#         for sample_size, demo_population in zip(n_sample_per_demo, population_per_demo)
#     ]
#     return pd.concat(poll_responders).reset_index(drop=True)
#
#
# def guaranteed_sample_pre_stratified(n_sample, shuffled_electorate, demographics, max_num_attempts):
#     """ie I'm going to call people at random, selected based on assumed demographic representation,
#     until I get 1000 responses"""
#     n_sample_per_demo = [
#         round(n_sample * demographic.population_percentage)
#         for demographic in demographics
#     ]
#     population_per_demo = [
#         shuffled_electorate.loc[demographic.population_segmentation.segment(shuffled_electorate), :]
#         for demographic in demographics
#     ]
#     # Convert these to one function that dependency injects the sampling function