import pandas as pd

from polling_simulator import aggregation


class TestNaiveAggregation:

    def test_works(self):
        data = pd.DataFrame({
            "candidate_preference": ["a", "b", "a", "b", "a"]
        })
        aggregate = aggregation.naive_aggregation(None, data, None).sort_values()
        expected_aggregate = pd.Series([3, 2], index=["a", "b"], name="candidate_preference").sort_values()
        pd.testing.assert_series_equal(aggregate, expected_aggregate)

class TestStratifiedAggregation:
    pass