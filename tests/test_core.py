import numpy as np
import pandas as pd

from polling_simulator import core


class TestVariable:

    def test_instantiates_ok(self):
        var = core.Variable("woo", lambda x: np.ones(x))
        assert var.name == "woo"


class TestSegmentationVariable:
    def test_general_working(self):
        var1 = core.Variable("var1", lambda x: np.ones(x))
        var2 = core.Variable("var2", lambda x: np.ones(x))
        var3 = core.Variable("var3", lambda x: np.ones(x))

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
        var = core.Variable("var", lambda x: np.ones(x))
        seg = (var >= 3)
        data = pd.DataFrame({"var": [1, 2, 3, 4, 5]})
        segment_mask = seg.segment(data)
        pd.testing.assert_series_equal(
            segment_mask,
            pd.Series([False, False, True, True, True], name="var")
        )

    def test_multiple_segments(self):
        var1 = core.Variable("var1", lambda x: np.ones(x))
        var2 = core.Variable("var2", lambda x: np.ones(x))
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
        var1 = core.Variable("var1", lambda x: np.ones(x))
        var2 = core.Variable("var2", lambda x: np.ones(x))
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

class TestSegmentationStr:
    def test_works_complex_case(self):
        var1 = core.Variable("var1", lambda x: np.ones(x))
        var2 = core.Variable("var2", lambda x: np.ones(x))

        seg1 = var1 >= 4
        seg2 = var2 != var1
        seg3 = seg2 == False
        final_seg = seg1 & (seg2 | seg3)
        assert str(final_seg) == "(var1 >= 4) & ((var2 != var1) | ((var2 != var1) == False))"