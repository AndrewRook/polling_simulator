import numpy as np
import pandas as pd


def naive_sample_predefined():
    """ie "I'm going to call 1000 people at total random, and we'll just
    use those responses as if they were representative"
    """


def naive_sample_guaranteed():
    """ie "I'm going to call people at random until I get 1000 responses,
    and we'll just use those responses as if they were representative"
    """


def pre_stratified_sample_predefined():
    """ie "I'm going to call 1000 people randomly selected to be based on
    assumed demographic representation, and we'll use those as representative"
    """


def pre_stratified_sample_guaranteed():
    """ie "I'm going to call people at random, selected based on assumed demographic
    representation, until I get 1000 responses, and we'll use those as representative"
    """


def post_stratified_sample_predefined():
    """ie "I'm going to call 1000 people at random, and then we'll weight the responses
    we get by assumed demographic representation
    """


def post_stratified_sample_guaranteed():
    """ie "I'm going to call people at random until I get 1000 responses, and then we'll
    weight the responses we get by assumed demographic representation
    """


def predefined_sample(n_sample, shuffled_electorate, demographics):
    """ie I'm going to call 1000 people at random, and just use those who answer"""
    if n_sample > len(shuffled_electorate):
        raise ValueError(f"number of samples ({n_sample}) greater than electorate ({len(shuffled_electorate)})")
    people_called = shuffled_electorate.head(n_sample)
    does_respond = people_called["response_likelihood"] > np.random.random(len(people_called))
    poll_responders = people_called[does_respond].reset_index(drop=True)
    return poll_responders


def guaranteed_sample(n_sample, shuffled_electorate, demographics):
    """ie I'm going to call people at random until I get 1000 responses"""
    does_respond = shuffled_electorate["response_likelihood"] > np.random.random(len(shuffled_electorate))
    poll_responders = shuffled_electorate[does_respond]
    if n_sample > len(poll_responders):
        raise ValueError(
            f"number of samples ({n_sample}) greater than number of poll responders ({len(poll_responders)})"
        )
    return poll_responders.head(n_sample)


def predefined_sample_pre_stratified(n_sample, shuffled_electorate, demographics):
    """ie I'm going to call 1000 people, randomly selected based on assumed demographic representation"""
    n_sample_per_demo = [
        round(n_sample * demographic.population_percentage)
        for demographic in demographics
    ]
    population_per_demo = [
        shuffled_electorate.loc[demographic.population_segmentation.segment(shuffled_electorate), :]
        for demographic in demographics
    ]
    poll_responders = [
        predefined_sample(sample_size, demo_population, None)
        for sample_size, demo_population in zip(n_sample_per_demo, population_per_demo)
    ]
    return pd.concat(poll_responders).reset_index(drop=True)


def guaranteed_sample_pre_stratified(n_sample, shuffled_electorate, demographics):
    """ie I'm going to call people at random, selected based on assumed demographic representation,
    until I get 1000 responses"""
    n_sample_per_demo = [
        round(n_sample * demographic.population_percentage)
        for demographic in demographics
    ]
    population_per_demo = [
        shuffled_electorate.loc[demographic.population_segmentation.segment(shuffled_electorate), :]
        for demographic in demographics
    ]
    # Convert these to one function that dependency injects the sampling function