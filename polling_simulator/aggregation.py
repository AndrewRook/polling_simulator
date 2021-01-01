import numpy as np
import pandas as pd

from copy import deepcopy
from typing import Iterable


def _no_turnout_weighting():
    def _weighting(poll_responses):
        return np.ones(len(poll_responses))
    return _weighting


def weight_by_self_reported_turnout(response_modifier=lambda x: x):
    """
    Weight the results based on the vote likelihood directly obtained from
    the poll respondents, optionally modified by a simple function.

    Parameters
    ----------
    response_modifier (optional): A function which takes in a pandas Series and returns a new Series.
        If set, the function will be used to apply a correction to the reported turnout (e.g. to
        downweight responders that claim to be 100% likely to vote).

    Returns
    -------
    The ``response_modifier`` function, wrapped to be used on the turnout likelihoods.
    """
    def _weighting(poll_responses):
        modified_turnout_likelihood = response_modifier(poll_responses["turnout_likelihood"])
        return modified_turnout_likelihood
    return _weighting


def weight_by_assumed_turnout(assumed_demographics: Iterable["Demographic"]):
    """
    Weight by the turnout likelihood assumed by a demographic model (rather than
    the actual reported likelihood as in ``weight_by_self_reported_turnout``).
    """
    raise NotImplementedError("Not yet implemented")


def naive_aggregation(turnout_weighting=_no_turnout_weighting()):
    """
    Sum up the candidate preferences of poll responders without regard to any demographic
    information.

    Parameters
    ----------
    turnout_weighting (optional): If set, apply a turnout weighting model before
        aggregation.

    Returns
    -------
    A function which takes in poll responses, does any necessary weighting, then
    sums up the (weighted) support for each candidate and returns the results as a pandas
    DataFrame

    """

    def _aggregation(poll_responses, poll_nonresponses):
        weighted_poll_responses = pd.DataFrame({
            "candidate_preference": poll_responses["candidate_preference"].values,
            "weight": turnout_weighting(poll_responses)
        })
        candidate_votes = weighted_poll_responses.groupby("candidate_preference")["weight"].sum()
        return candidate_votes
    return _aggregation


def stratified_aggregation(
        assumed_demographics: Iterable["Demographic"],
        population_fraction_per_demographic: Iterable[float],
        turnout_weighting=_no_turnout_weighting()):
    if abs(sum(population_fraction_per_demographic) - 1) > 1e-4:
        raise ValueError(f"demographic populations do not sum to 1: {population_fraction_per_demographic}")

    assumed_demographics = deepcopy(assumed_demographics)
    population_fraction_per_demographic = deepcopy(population_fraction_per_demographic)

    def _aggregation(poll_responses, _):
        stratified_votes = []
        responses_in_demographic = [
            demographic.population_segmentation.segment(poll_responses)
            for demographic in assumed_demographics
        ]
        num_responses_per_demographic = [
            population.sum()
            for population in responses_in_demographic
        ]
        if sum(num_responses_per_demographic) != len(poll_responses):
            raise ValueError(f"""
Demographics are not mutually exclusive and completely exhaustive. {len(poll_responses)}
poll responders were split into demographic groups totaling {sum(num_responses_per_demographic)}.
        """)
        for responses_in_demographic, num_responses, population_fraction in zip(
                responses_in_demographic, num_responses_per_demographic, population_fraction_per_demographic
        ):
            raw_votes = naive_aggregation(turnout_weighting)(poll_responses[responses_in_demographic], None)
            stratified_votes.append(
                raw_votes # raw aggregation of polled people in the demographic
                * population_fraction # the bigger the demo is in the whole population, the higher the weight
                / (num_responses / len(poll_responses)) # relative to the prevalence of the demo in the poll
            )
            # Don't need to worry about the denominator of the weights since we'll scale to
            # percentages anyway

        stratified_votes = pd.concat(stratified_votes).reset_index().groupby("candidate_preference")["weight"].sum()
        return stratified_votes
    return _aggregation


"""
* likely voter weighting in aggregation
   - probabilistic weighting based on either actual voting likelihood or demographic assumptions
"""