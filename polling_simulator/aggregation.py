import numpy as np
import pandas as pd

from copy import deepcopy


def _no_turnout_weighting():
    def _weighting(poll_responses):
        return np.ones(len(poll_responses))
    return _weighting


def naive_aggregation(turnout_weighting=_no_turnout_weighting()):

    def _aggregation(poll_responses, poll_nonresponses):
        weighted_poll_responses = pd.DataFrame({
            "candidate_preference": poll_responses["candidate_preference"].values,
            "weight": turnout_weighting(poll_responses)
        })
        candidate_votes = weighted_poll_responses.groupby("candidate_preference")["weight"].sum()
        return candidate_votes
    return _aggregation


def stratified_aggregation(assumed_demographics, turnout_weighting=_no_turnout_weighting()):
    assumed_demographics = deepcopy(assumed_demographics)

    def _aggregation(poll_responses, poll_nonresponses):
        stratified_votes = []
        for demographic in assumed_demographics:
            responses_in_demographic = demographic.population_segmentation.segment(poll_responses)
            raw_votes = naive_aggregation(turnout_weighting)(poll_responses[responses_in_demographic], poll_nonresponses)
            population_weight = demographic.population_percentage
            stratified_votes.append(
                raw_votes # raw aggregation of polled people in the demographic
                * population_weight # the size of the demographic
                * len(poll_responses) # the relative size of the demographic in the poll compared to all poll responses
                / responses_in_demographic.sum())
            # Don't need to worry about the denominator of the weights since we'll scale to
            # percentages anyway

        stratified_votes = pd.concat(stratified_votes).reset_index().groupby("candidate_preference")["weight"].sum()
        return stratified_votes
    return _aggregation


"""
* likely voter weighting in aggregation
   - probabilistic weighting based on either actual voting likelihood or demographic assumptions
"""