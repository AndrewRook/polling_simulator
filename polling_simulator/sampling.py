import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Callable, Union, Tuple


def predefined_sample(max_num_attempts, screen_likely_voters):
    """
    Generate a function that will contact a particular number of people and use whoever responds.
    (i.e. "I'm going to call 1000 people at random, and just use those who answer".)

    Parameters
    ----------
    max_num_attempts: Number of times to attempt to contact each potential participant.
    screen_likely_voters: If ``True``, proportionately remove participants based on their turnout likelihood
        (e.g. if someone has a turnout likelihood of 0.7, then they have a 30% chance of being rejected by the
        sampler.

    Returns
    -------
    A sampling function, to be used as ``sampling_strategy`` in ``simulate.run_polls``
    or ``sampling.stratified_sample``.
    """
    def _sampler(n_sample, shuffled_electorate):
        if n_sample > len(shuffled_electorate):
            raise ValueError(f"number of samples ({n_sample}) greater than electorate ({len(shuffled_electorate)})")
        people_called = shuffled_electorate.head(n_sample)
        does_respond, num_attempts_required = _get_responses(
            people_called["response_likelihood"], max_num_attempts
        )
        likely_voter = (
            np.ones(len(people_called)).astype(bool) if screen_likely_voters == False
            else np.random.random(len(people_called)) < people_called["turnout_likelihood"]
        )
        poll_responders = people_called[does_respond & likely_voter].reset_index(drop=True)
        poll_responders["num_contact_attempts"] = num_attempts_required[does_respond & likely_voter]
        poll_non_responders = people_called[does_respond == False].reset_index(drop=True)
        poll_non_responders["num_contact_attempts"] = max_num_attempts
        if len(poll_responders) == 0:
            raise ValueError("Poll returned no valid responses")
        return poll_responders, poll_non_responders
    return _sampler


def guaranteed_sample(max_num_attempts, screen_likely_voters):
    """
    Generate a function that will contact people until they get a specific number of "good" responses, thus
    guaranteeing a specific sample size.
    (i.e. "I'm going to call people at random until I get 1000 responses".)

    Parameters
    ----------
    max_num_attempts: Number of times to attempt to contact each potential participant.
    screen_likely_voters: If ``True``, proportionately remove participants based on their turnout likelihood
        (e.g. if someone has a turnout likelihood of 0.7, then they have a 30% chance of being rejected by the
        sampler.

    Returns
    -------
    A sampling function, to be used as ``sampling_strategy`` in ``simulate.run_polls``
    or ``sampling.stratified_sample``.
    """
    def _sampler(n_sample, shuffled_electorate):
        does_respond, num_attempts_required = _get_responses(
            shuffled_electorate["response_likelihood"], max_num_attempts
        )
        likely_voter = (
            np.ones(len(shuffled_electorate)).astype(bool) if screen_likely_voters == False
            else (np.random.random(len(shuffled_electorate)) < shuffled_electorate["turnout_likelihood"]).values
        )
        cumulative_valid_responses = np.cumsum(does_respond & likely_voter)
        #breakpoint()
        if cumulative_valid_responses[-1] < n_sample:
            raise ValueError(
                f"number of samples ({n_sample}) greater than number of valid poll responders ({cumulative_valid_responses[-1]})"
            )
        people_contacted = shuffled_electorate[cumulative_valid_responses <= n_sample]
        does_respond = does_respond[cumulative_valid_responses <= n_sample]
        likely_voter = likely_voter[cumulative_valid_responses <= n_sample]
        num_attempts_required = num_attempts_required[cumulative_valid_responses <= n_sample]
        poll_responders = people_contacted.loc[does_respond & likely_voter, :].reset_index(drop=True)
        poll_responders["num_contact_attempts"] = num_attempts_required[does_respond & likely_voter]
        poll_nonresponders = people_contacted.loc[does_respond == False, :].reset_index(drop=True)
        poll_nonresponders["num_contact_attempts"] = max_num_attempts

        return poll_responders, poll_nonresponders
    return _sampler


def stratified_sample(
        assumed_demographics,
        sampling_strategy: Callable
):
    """
    Generate a function that will proportionately sample people based on their demographics.
    For instance, if you have a demographic containing only 10% of the total electorate, this
    function will attempt to fill out your sample to contain 10% of the demographic.

    How close to that 10% you get will depend on the sampling strategy you pass in. For example,
    if you want 10% of a population with a really low response rate and you use the ``predefined_sample``
    sampler, you may get much less than 10% in your final sample.

    Parameters
    ----------
    assumed_demographics: A list of all the demographics you want to stratify by.
    sampling_strategy: A function that represents the core sampling strategy you want to use on each
        demographic (e.g., the output of ``guaranteed_sample``).

    Returns
    -------
    A sampling function, to be used as ``sampling_strategy`` in ``simulate.run_polls``.
    """
    # It's critical to make a copy of the demographics here, otherwise
    # they can be mutated outside of the closure!
    assumed_demographics = deepcopy(assumed_demographics)

    def _sampler(n_sample, shuffled_electorate):

        population_per_demographic = [
            shuffled_electorate.loc[
                demographic.population_segmentation.segment(shuffled_electorate), :
            ].copy(deep=True)
            for demographic in assumed_demographics
        ]
        population_in_all_demographics = sum([
            len(population)
            for population in population_per_demographic
        ])
        if population_in_all_demographics != len(shuffled_electorate):
            raise ValueError(f"""
Demographics are not mutually exclusive and completely exhaustive. A {len(shuffled_electorate)}-person
electorate was split into demographic groups totaling {population_in_all_demographics}.
        """)

        n_sample_per_demographic = [
            round(n_sample * len(population) / population_in_all_demographics)
            for population in population_per_demographic
        ]
        poll_responders, poll_nonresponders = list(zip(*[
            sampling_strategy(sample_size, demo_population)
            for sample_size, demo_population in zip(n_sample_per_demographic, population_per_demographic)
        ]))
        return pd.concat(poll_responders).reset_index(drop=True), pd.concat(poll_nonresponders).reset_index(drop=True)
    return _sampler


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
