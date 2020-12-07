import numpy as np

from functools import partial
from scipy import stats

from polling_simulator import Variable, Demographic
from polling_simulator import aggregation, sampling
from polling_simulator.distributions import truncated_gaussian_distribution
from polling_simulator import generate_electorate, run_elections, run_polls


if __name__ == "__main__":
    everyone = Variable("everyone", lambda x: np.ones(x).astype(bool))

    demographics = [
        Demographic(
            population_percentage=1.0,
            turnout_likelihood=1.0,
            response_likelihood=1.0,
            candidate_preference={"Dem": 0.51, "Rep": 0.47, "Ind": 1 - 0.51 - 0.47},
            population_segmentation=(everyone == True)
        )
    ]
    electorate = generate_electorate(15500000, demographics)
    breakpoint()
    # np.random.seed(123)
    # age = Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))
    # gender = Variable("gender", partial(
    #     np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.49, 0.51])
    # ))
    # young_men = Demographic(
    #     0.25,
    #     0.5,
    #     0.1,
    #     {"a": 0.5, "b": 0.5},
    #     (age < 40) & (gender == "M")
    # )
    # old_men = Demographic(
    #     0.25,
    #     0.7,
    #     0.2,
    #     {"b": 1},
    #     (age >= 40) & (gender == "M")
    # )
    # young_women = Demographic(
    #     0.25,
    #     0.6,
    #     0.05,
    #     {"a": 1},
    #     (age < 40) & (gender == "F")
    # )
    # old_women = Demographic(
    #     0.25,
    #     0.8,
    #     0.2,
    #     {"a": 1},
    #     (age >= 40) & (gender == "F")
    # )
    # np.random.seed(123)
    # electorate = generate_electorate(
    #     100000,
    #     [
    #         young_men, old_men, young_women, old_women
    #     ]
    # )
    # polls = run_polls(
    #     10,
    #     1000, electorate, [young_men, old_men, young_women, old_women],
    #     sampling.guaranteed_sample(1, False), aggregation.naive_aggregation()
    # )
    # results = run_elections(10, electorate)
    breakpoint()
