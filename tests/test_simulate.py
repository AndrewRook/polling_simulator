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

# class TestInternalGenerateDemographicPopulation:
#
#     def test_appropriately_excludes_data(self, gender, age):
#         test_demo = Demographic(
#             1, 1, 1, {"a": 1},
#             (gender == "M") & (age < 40)
#         )
#         np.random.seed(123)
#         population = simulate._generate_demographic_population(100, test_demo, [age, gender], ["a"])
#         assert len(population) == 100
#         assert population["age"].max() < 40
#         np.testing.assert_array_equal(population["gender"].unique(), np.array(["M"]))
#
#     def test_appropriately_samples_candidate_preference(self, gender):
#         test_demo = Demographic(
#             1, 1, 1, {"a": 0.4, "b": 0.6},
#             (gender == "M") | (gender == "F")
#         )
#         np.random.seed(123)
#         population = simulate._generate_demographic_population(100000, test_demo, [gender], ["a", "b"])
#
#         assert len(population) == 100000
#
#         assert abs(60000 - (population["candidate_preference"] == "b").sum()) < 500
#         assert abs(40000 - (population["candidate_preference"] == "a").sum()) < 500


class TestRunElection:
    def test_applies_turnout_correctly(self, gender):
        low_turnout = Demographic(
            0.5, 0.1, 1, {"a": 1},
            (gender == "M") | (gender == "F")
        )
        high_turnout = Demographic(
            0.5, 0.9, 1, {"b": 1},
            (gender == "M") | (gender == "F")
        )
        np.random.seed(123)
        electorate = simulate.generate_electorate(
            20000, [low_turnout, high_turnout]
        )
        result = simulate.run_election(electorate)
        assert abs(1000 - result["a"]) < 50
        assert abs(9000 - result["b"]) < 50


class TestRunMultipleElections:
    def test_handles_low_vote_candidates(self, gender):
        low_turnout = Demographic(
            0.05, 0.01, 1, {"a": 1},
            (gender == "M") | (gender == "F")
        )
        high_turnout = Demographic(
            0.95, 0.9, 1, {"b": 1},
            (gender == "M") | (gender == "F")
        )
        np.random.seed(123)
        electorate = simulate.generate_electorate(
            2000, [low_turnout, high_turnout]
        )
        results = simulate.run_elections(10, electorate)
        assert "a" in results.columns
        assert results["a"].min() == 0
        assert results.dtypes["a"] == np.int
