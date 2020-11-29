import numpy as np
import pandas as pd
import pytest

from functools import partial

from polling_simulator import sampling
from polling_simulator import Variable, Demographic
from polling_simulator.distributions import truncated_gaussian_distribution


class TestPredefinedSample:
    def test_fails_when_asked_for_too_many_people(self):
        data = pd.DataFrame({
            "response_likelihood": np.ones(10)
        })
        with pytest.raises(ValueError):
            sampling.predefined_sample(1, False)(20, data)

    def test_fails_when_response_likelihood_is_too_low(self):
        data = pd.DataFrame({
            "response_likelihood": np.ones(1000) * 0.000001
        })
        with pytest.raises(ValueError):
            sampling.predefined_sample(1, False)(20, data)

    def test_fails_when_turnout_likelihood_is_too_low(self):
        data = pd.DataFrame({
            "response_likelihood": np.ones(1000),
            "turnout_likelihood": np.ones(1000) * 0.000001
        })
        sampling.predefined_sample(1, False)(500, data)
        with pytest.raises(ValueError):
            sampling.predefined_sample(1, True)(500, data)

    def test_returns_fewer_than_desired_when_response_rates_are_low(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(20) * 0.5
        })
        poll_responders, poll_non_responders = sampling.predefined_sample(1, False)(10, data)
        assert len(poll_responders) < 10
        assert len(poll_responders) + len(poll_non_responders) == 10

    def test_reaches_more_people_when_makes_multiple_attempts(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(40) * 0.1
        })
        single_call_responders, _ = sampling.predefined_sample(1, False)(20, data)
        multi_call_responders, non_responders = sampling.predefined_sample(5, False)(20, data)
        assert len(multi_call_responders) > len(single_call_responders)
        assert len(multi_call_responders) < 20
        assert len(multi_call_responders) + len(non_responders) == 20

    def test_applies_likely_voter_screen_correctly(self):
        data = pd.DataFrame({
            "response_likelihood": np.ones(1000),
            "turnout_likelihood": np.ones(1000) * 0.1
        })
        responders, nonresponders = sampling.predefined_sample(1, True)(50, data)
        assert len(responders) + len(nonresponders) < 50


class TestGuaranteedSample:
    def test_fails_when_asked_for_too_many_people(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10) * 0.05
        })
        with pytest.raises(ValueError):
            sampling.guaranteed_sample(1, False)(5, data)

    @pytest.mark.parametrize("num_people", [10, 50, 100, 500])
    def test_always_returns_the_asked_for_number_of_people(self, num_people):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10000) * 0.1
        })
        responders, non_responders = sampling.guaranteed_sample(1, False)(num_people, data)
        assert len(responders) == num_people
        assert len(non_responders) > 0

    def test_works_with_multiple_attempts(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10000) * 0.1
        })
        single_attempt_responders, single_attempt_non_responders = sampling.guaranteed_sample(1, False)(100, data)
        multiple_attempt_responders, multiple_attempt_non_responders = sampling.guaranteed_sample(5, False)(100, data)

        assert len(single_attempt_responders) == len(multiple_attempt_responders)
        assert len(multiple_attempt_non_responders) < len(single_attempt_non_responders)

    def test_works_with_likely_voter_screen(self):
        np.random.seed(123)
        data = pd.DataFrame({
            "response_likelihood": np.ones(10000) * 0.5,
            "turnout_likelihood": np.ones(10000) * 0.1
        })
        _, no_screen_non_responders = sampling.guaranteed_sample(1, False)(200, data)
        responders, screened_non_responders = sampling.guaranteed_sample(1, True)(200, data)
        assert len(responders) == 200
        assert len(screened_non_responders) > 5 * len(no_screen_non_responders)


class TestPreStratifiedSample:
    def test_freezes_assumed_demographics(self):
        age = Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))
        young_people = Demographic(0.5, 0.5, 0.1, {"a": 1}, age < 40)
        old_people = Demographic(0.5, 0.5, 0.1, {"b": 1}, age >= 40)
        demographics = [
            young_people, old_people
        ]
        sampler = sampling.stratified_sample(demographics, sampling.guaranteed_sample(1, False))
        demographics.pop(0)
        assert len(demographics) == 1
        for item in sampler.__closure__:
            if type(item.cell_contents) == list:
                assert len(item.cell_contents) == 2

    def test_stratification_works_as_expected(self):
        np.random.seed(123)
        gender = Variable("gender", partial(
            np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.5, 0.5])
        ))

        men = Demographic(0.5, 0.5, 0.1, {"a": 1}, (gender == "M"))
        women = Demographic(0.5, 0.5, 0.2, {"b": 1}, (gender == "F"))
        demographics = [men, women]
        sampler = sampling.stratified_sample(demographics, sampling.guaranteed_sample(1, False))

        male_electorate = pd.DataFrame({
            "turnout_likelihood": np.ones(50000) * men.turnout_likelihood,
            "response_likelihood": men.response_likelihood,
            "candidate_preference": "a",
            "gender": "M"
        })
        female_electorate = pd.DataFrame({
            "turnout_likelihood": np.ones(50000) * women.turnout_likelihood,
            "response_likelihood": women.response_likelihood,
            "candidate_preference": "b",
            "gender": "F"
        })
        shuffled_electorate = pd.concat([male_electorate, female_electorate]).sample(frac=1)

        responders, non_responders = sampler(1000, shuffled_electorate)
        assert np.sum(responders["gender"] == "F") == 500
        assert np.sum(responders["gender"] == "M") == 500
        assert len(non_responders) > 3000
        assert np.sum(non_responders["gender"] == "M") > np.sum(non_responders["gender"] == "F")


class TestInternalGetResponses:
    def test_fails_when_passed_zero_attempts(self):
        with pytest.raises(ValueError):
            sampling._get_responses(None, 0)

    def test_works_with_single_attempt(self):
        np.random.seed(123)
        response_likelihoods = np.ones(40) * 0.5
        did_respond, num_attempts_required = sampling._get_responses(response_likelihoods, 1)
        assert np.sum(did_respond) < len(response_likelihoods)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1]))
        assert np.min(num_attempts_required[did_respond]) == 1

    def test_works_with_pandas_series(self):
        np.random.seed(123)
        response_likelihoods = pd.Series(np.ones(40) * 0.5)
        did_respond, num_attempts_required = sampling._get_responses(response_likelihoods, 1)
        assert np.sum(did_respond) < len(response_likelihoods)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1]))
        assert np.min(num_attempts_required[did_respond]) == 1

    def test_works_with_multiple_attempts(self):
        np.random.seed(123)
        response_likelihoods = np.ones(40) * 0.25
        did_respond_single, _ = sampling._get_responses(response_likelihoods, 1)
        did_respond_multi, num_attempts_required = sampling._get_responses(response_likelihoods, 3)
        assert np.sum(did_respond_multi) > np.sum(did_respond_single)
        np.testing.assert_allclose(np.unique(num_attempts_required), np.array([-1, 1, 2, 3]))
        assert np.min(num_attempts_required[did_respond_multi]) == 1
