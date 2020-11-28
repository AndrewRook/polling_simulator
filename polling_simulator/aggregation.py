import pandas as pd


def naive_aggregation(assumed_demographics, poll_responses, poll_nonresponses):
    candidate_votes = poll_responses["candidate_preference"].value_counts()
    return candidate_votes


def stratified_aggregation(assumed_demographics, poll_responses, poll_nonresponses):
    stratified_votes = []
    for demographic in assumed_demographics:
        responses_in_demographic = demographic.population_segmentation.segment(poll_responses)
        raw_votes = naive_aggregation(None, poll_responses[responses_in_demographic], None)
        weight = demographic.population_percentage * demographic.turnout_likelihood
        stratified_votes.append(raw_votes * weight)
        # Don't need to worry about the denominator of the weights since we'll scale to
        # percentages anyway

    stratified_votes = pd.concat(stratified_votes).reset_index().groupby("index").sum()
    return stratified_votes

