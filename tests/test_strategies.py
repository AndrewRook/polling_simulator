import numpy as np
import pandas as pd
import pytest

from polling_simulator import strategies

# @pytest.fixture(scope="function")
# def electorate():
#     data = pd.DataFrame({
#         "response_likelihood": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#     })


class TestPredefinedSample:
    def test_fails_when_asked_for_too_many_people(self):
        data = pd.DataFrame({
            "response_likelihood": np.ones(10)
        })
        with pytest.raises(ValueError):
            strategies.predefined_sample(20, data, 1)

    def test_returns_fewer_than_desired_when_response_rates_are_low(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(20) * 0.5
        })
        poll_responders, poll_non_responders = strategies.predefined_sample(10, data, 1)
        assert len(poll_responders) < 10
        assert len(poll_responders) + len(poll_non_responders) == 10

    def test_reaches_more_people_when_makes_multiple_attempts(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(40) * 0.5
        })
        single_call_responders, _ = strategies.predefined_sample(20, data, 1)
        multi_call_responders, non_responders = strategies.predefined_sample(20, data, 5)
        assert len(multi_call_responders) > len(single_call_responders)
        assert len(multi_call_responders) < 20
        assert len(multi_call_responders) + len(non_responders) == 20


class TestGuaranteedSample:
    def test_fails_when_asked_for_too_many_people(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10) * 0.05
        })
        with pytest.raises(ValueError):
            strategies.guaranteed_sample(5, data, 1)

    @pytest.mark.parametrize("num_people", [10, 50, 100, 500])
    def test_always_returns_the_asked_for_number_of_people(self, num_people):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10000) * 0.1
        })
        responders, non_responders = strategies.guaranteed_sample(num_people, data, 1)
        assert len(responders) == num_people
        assert len(non_responders) > 0

    def test_works_with_multiple_attempts(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10000) * 0.1
        })
        single_attempt_responders, single_attempt_non_responders = strategies.guaranteed_sample(100, data, 1)
        multiple_attempt_responders, multiple_attempt_non_responders = strategies.guaranteed_sample(100, data, 5)

        assert len(single_attempt_responders) == len(multiple_attempt_responders)
        assert len(multiple_attempt_non_responders) < len(single_attempt_non_responders)


class TestInternalGetResponses:
    def test_fails_when_passed_zero_attempts(self):
        with pytest.raises(ValueError):
            strategies._get_responses(None, 0)

    def test_works_with_single_attempt(self):
        np.random.seed(123)
        response_likelihoods = np.ones(40) * 0.5
        did_respond, num_attempts_required = strategies._get_responses(response_likelihoods, 1)
        assert np.sum(did_respond) < len(response_likelihoods)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1]))
        assert np.min(num_attempts_required[did_respond]) == 1

    def test_works_with_pandas_series(self):
        np.random.seed(123)
        response_likelihoods = pd.Series(np.ones(40) * 0.5)
        did_respond, num_attempts_required = strategies._get_responses(response_likelihoods, 1)
        assert np.sum(did_respond) < len(response_likelihoods)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1]))
        assert np.min(num_attempts_required[did_respond]) == 1

    def test_works_with_multiple_attempts(self):
        np.random.seed(123)
        response_likelihoods = np.ones(40) * 0.25
        did_respond_single, _ = strategies._get_responses(response_likelihoods, 1)
        did_respond_multi, num_attempts_required = strategies._get_responses(response_likelihoods, 3)
        assert np.sum(did_respond_multi) > np.sum(did_respond_single)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1, 2, 3]))
        assert np.min(num_attempts_required[did_respond_multi]) == 1
