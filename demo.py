import numpy as np

from functools import partial
from scipy import stats

from polling_simulator import Variable, Demographic
from polling_simulator.distributions import truncated_gaussian_distribution
from polling_simulator import generate_electorate, run_multiple_elections, run_poll


if __name__ == "__main__":
    np.random.seed(123)
    age = Variable("age", truncated_gaussian_distribution(25, 25, 18, 110))
    gender = Variable("gender", partial(
        np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.49, 0.51])
    ))
    young_men = Demographic(
        0.25,
        0.5,
        0.1,
        {"a": 0.5, "b": 0.5},
        (age < 40) & (gender == "M")
    )
    old_men = Demographic(
        0.25,
        0.7,
        0.2,
        {"b": 1},
        (age >= 40) & (gender == "M")
    )
    young_women = Demographic(
        0.25,
        0.6,
        0.05,
        {"a": 1},
        (age < 40) & (gender == "F")
    )
    old_women = Demographic(
        0.25,
        0.8,
        0.2,
        {"a": 1},
        (age >= 40) & (gender == "F")
    )
    np.random.seed(123)
    electorate = generate_electorate(
        100000,
        [
            young_men, old_men, young_women, old_women
        ]
    )
    from polling_simulator import aggregation, sampling
    poll = run_poll(
        1000, electorate, [young_men, old_men, young_women, old_women],
        sampling.guaranteed_sample(1), aggregation.stratified_aggregation
    )
    results = run_multiple_elections(10, electorate)
    breakpoint()
