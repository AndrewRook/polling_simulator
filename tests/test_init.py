import numpy as np
import pandas as pd
import pytest

from functools import partial

import polling_simulator as ps


class TestVariable:

    def test_instantiates_ok(self):
        var = ps.Variable("woo", lambda x: np.ones(x))
        assert var.name == "woo"


class TestSegmentationVariable:
    def test_general_working(self):
        var1 = ps.Variable("var1", lambda x: np.ones(x))
        var2 = ps.Variable("var2", lambda x: np.ones(x))
        var3 = ps.Variable("var3", lambda x: np.ones(x))

        seg = (
            ((var1 > 3) & (var2 == 5)) |
            (
                (var1 == 10) &
                ((var2 < var1) | (var3 > 5))
            )
        )
        seg_variables = seg.variables
        assert len(seg_variables) == 3
        assert seg_variables[0] is var1
        assert seg_variables[1] is var2
        assert seg_variables[2] is var3


class TestSegmentationSegment:
    def test_general_working(self):
        var = ps.Variable("var", lambda x: np.ones(x))
        seg = (var >= 3)
        data = pd.DataFrame({"var": [1, 2, 3, 4, 5]})
        segment_mask = seg.segment(data)
        pd.testing.assert_series_equal(
            segment_mask,
            pd.Series([False, False, True, True, True], name="var")
        )

    def test_multiple_segments(self):
        var1 = ps.Variable("var1", lambda x: np.ones(x))
        var2 = ps.Variable("var2", lambda x: np.ones(x))
        seg1 = var1 >= 3
        seg2 = var2 < 5
        seg = seg1 & seg2
        data = pd.DataFrame({
            "var1": [1, 2, 3, 4, 5],
            "var2": [1, 5, 1, 5, 1]
        })

        segment_mask = seg.segment(data)
        pd.testing.assert_series_equal(
            segment_mask,
            pd.Series([False, False, True, False, True])
        )

    def test_order_of_operation(self):
        data = pd.DataFrame({
            "var1": [1, 2, 3, 4, 5],
            "var2": [1, 5, 1, 5, 1]
        })
        var1 = ps.Variable("var1", lambda x: np.ones(x))
        var2 = ps.Variable("var2", lambda x: np.ones(x))
        seg1 = var1 >= 4
        seg2 = var2 < 5
        seg3 = (var1 == 2)

        seg_explicit_order = (seg3 | seg1) & seg2
        segment_explicit_order_mask = seg_explicit_order.segment(data)
        pd.testing.assert_series_equal(
            segment_explicit_order_mask,
            pd.Series([False, False, False, False, True])
        )

        seg_implicit_order = seg3 | seg1 & seg2
        segment_implicit_order_mask = seg_implicit_order.segment(data)
        pd.testing.assert_series_equal(
            segment_implicit_order_mask,
            pd.Series([False, True, False, False, True])
        )


class TestInternalGenerateDemographicPopulation:
    @pytest.fixture(scope="function")
    def age(self):
        return ps.Variable("age", ps.truncated_gaussian_distribution(25, 25, 18, 110))

    @pytest.fixture(scope="function")
    def gender(self):
        return ps.Variable("gender", partial(
        np.random.choice, np.array(["M", "F"]), replace=True, p=np.array([0.49, 0.51])
    ))

    def test_appropriately_excludes_data(self, gender, age):
        test_demo = ps.Demographic(
            1, 1, {"a": 1},
            (gender == "M") & (age < 40)
        )
        np.random.seed(123)
        population = ps._generate_demographic_population(100, test_demo, [age, gender])
        assert len(population) == 100
        assert population["age"].max() < 40
        np.testing.assert_array_equal(population["gender"].unique(), np.array(["M"]))

    def test_appropriately_samples_candidate_preference(self, gender):
        test_demo = ps.Demographic(
            1, 1, {"a": 0.4, "b": 0.6},
            (gender == "M") | (gender == "F")
        )
        np.random.seed(123)
        population = ps._generate_demographic_population(100000, test_demo, [gender])

        assert len(population) == 100000

        assert abs(60000 - (population["candidate_preference"] == "b").sum()) < 500
        assert abs(40000 - (population["candidate_preference"] == "a").sum()) < 500

