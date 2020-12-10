import numpy as np
import pandas as pd
import pytest

from functools import partial

from polling_simulator.core import Demographic, Variable
from polling_simulator.distributions import truncated_gaussian_distribution
from polling_simulator import simulate


@pytest.fixture(scope="function")
def age():
    return Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))


@pytest.fixture(scope="function")
def gender():
    return Variable("gender", partial(
        np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.49, 0.51])
    ))


class TestGenerateElectorate:
    def test_works_in_normal_case(self, gender):
        np.random.seed(123)

        demographics = [
            Demographic(1, 1, {"a": 1}, gender == "M"),
            Demographic(0.5, 0.4, {"a": 0.25, "b": 0.75}, gender == "F")
        ]
        electorate = simulate.generate_electorate(10000, demographics)
        assert len(electorate) == 10000
        assert abs((electorate["gender"] == "M").sum() / len(electorate) - 0.49) < 1e-2
        assert abs((electorate["gender"] == "F").sum() / len(electorate) - 0.51) < 1e-2
        assert abs(electorate["turnout_likelihood"].mean() - (1 * 0.49 + 0.5 * 0.51)) < 1e-2
        assert abs(electorate["response_likelihood"].mean() - (1 * 0.49 + 0.4 * 0.51)) < 1e-2
        assert abs(
            (electorate["candidate_preference"] == "a").sum() / len(electorate) - (1 * 0.49 + 0.25 * 0.51)
        ) < 1e-2
        assert abs(
            (electorate["candidate_preference"] == "b").sum() / len(electorate) - (0.75 * 0.51)
        ) < 1e-2


class TestGenerateDemographicFeaturesOfPopulation:
    def test_fails_when_demographics_overlap(self, age):
        np.random.seed(123)
        population = pd.DataFrame({
            "age": age.data_generator(10000)
        })
        demographics = [
            Demographic(1, 1, {"a": 1}, age < 40),
            Demographic(1, 1, {"b": 1}, age >= 35)
        ]
        with pytest.raises(ValueError):
            simulate.generate_demographic_features_of_population(population, demographics, ["a", "b"])

    def test_fails_when_demographics_miss_people(self, age):
        np.random.seed(123)
        population = pd.DataFrame({
            "age": age.data_generator(10000)
        })
        demographics = [
            Demographic(1, 1, {"a": 1}, age < 40),
            Demographic(1, 1, {"b": 1}, age > 45)
        ]
        with pytest.raises(ValueError):
            simulate.generate_demographic_features_of_population(population, demographics, ["a", "b"])

    def test_segments_appropriately(self, gender):
        np.random.seed(123)
        population = pd.DataFrame({
            "gender": gender.data_generator(10000)
        })
        demographics = [
            Demographic(1, 1, {"a": 1}, gender == "M"),
            Demographic(0.5, 0.4, {"a": 0.25, "b": 0.75}, gender == "F")
        ]
        demographic_features = simulate.generate_demographic_features_of_population(
            population, demographics, ["a", "b"]
        )
        assert abs(demographic_features["turnout_likelihood"].mean() - (1 * 0.49 + 0.5 * 0.51)) < 0.1
        assert abs(demographic_features["response_likelihood"].mean() - (1 * 0.49 + 0.4 * 0.51)) < 0.1
        assert abs(
            (demographic_features["candidate_preference"] == "a").sum() / len(demographic_features) -
            (1 * 0.49 + 0.25 * 0.51)
        ) < 0.1
        assert abs(
            (demographic_features["candidate_preference"] == "b").sum() / len(demographic_features) -
            (0.75 * 0.51)
        ) < 0.1


class TestRunElection:
    def test_applies_turnout_correctly(self, gender):
        low_turnout = Demographic(
            0.1, 1, {"a": 1},
            (gender == "M")
        )
        high_turnout = Demographic(
            0.9, 1, {"b": 1},
            (gender == "F")
        )
        np.random.seed(123)
        electorate = simulate.generate_electorate(
            20000, [low_turnout, high_turnout]
        )
        result = simulate.run_election(electorate)
        assert abs(20000 * 0.49 * 0.1 - result["a"]) < 100
        assert abs(20000 * 0.51 * 0.9 - result["b"]) < 100


class TestRunMultipleElections:
    def test_handles_low_vote_candidates(self, gender):
        low_turnout = Demographic(
            0.0001, 1, {"a": 1},
            (gender == "M")
        )
        high_turnout = Demographic(
            0.9, 1, {"b": 1},
            (gender == "F")
        )
        np.random.seed(123)
        electorate = simulate.generate_electorate(
            2000, [low_turnout, high_turnout]
        )
        results = simulate.run_elections(10, electorate)
        assert "a" in results.columns
        assert results["a"].min() == 0
        assert results.dtypes["a"] == np.int
