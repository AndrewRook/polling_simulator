import numpy as np
import pandas as pd

from functools import partial

from polling_simulator import aggregation, Variable, Demographic


class TestNaiveAggregation:

    def test_works(self):
        data = pd.DataFrame({
            "candidate_preference": ["a", "b", "a", "b", "a"]
        })
        aggregate = aggregation.naive_aggregation()(data, None).sort_values()
        expected_aggregate = pd.Series([3.0, 2.0], index=["a", "b"]).sort_values()
        pd.testing.assert_series_equal(aggregate, expected_aggregate, check_names=False)


class TestStratifiedAggregation:
    def test_works_no_weighting(self):
        gender = Variable("gender", partial(
            np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.5, 0.5])
        ))

        men = Demographic(0.5, 0.5, {"a": 1}, (gender == "M"))
        women = Demographic(0.5, 1, {"b": 1}, (gender == "F"))

        male_poll = pd.DataFrame({
            "turnout_likelihood": np.ones(1000) * men.turnout_likelihood,
            "response_likelihood": men.response_likelihood,
            "candidate_preference": "a",
            "gender": "M"
        })
        female_poll = pd.DataFrame({
            "turnout_likelihood": np.ones(2000) * women.turnout_likelihood,
            "response_likelihood": women.response_likelihood,
            "candidate_preference": "b",
            "gender": "F"
        })
        poll_results = pd.concat([male_poll, female_poll]).sample(frac=1)

        naive_aggregate = aggregation.naive_aggregation()(poll_results, None).sort_values()
        stratified_aggregate = aggregation.stratified_aggregation(
            [men, women], [0.5, 0.5]
        )(poll_results, None).sort_values()
        pd.testing.assert_series_equal(
            pd.Series([1000.0, 2000.0], index=["a", "b"]).sort_values(),
            naive_aggregate,
            check_names=False
        )
        pd.testing.assert_series_equal(
            pd.Series([1500.0, 1500.0], index=["a", "b"]).sort_values(),
            stratified_aggregate,
            check_names=False
        )
