import numpy as np
import pandas as pd

from polling_simulator import Variable, Segmentation


class TestVariable:

    def test_instantiates_ok(self):
        var = Variable("woo", lambda x: np.ones(x))
        assert var.name == "woo"


class TestSegmentationVariable:
    def test_general_working(self):
        var1 = Variable("var1", lambda x: np.ones(x))
        var2 = Variable("var2", lambda x: np.ones(x))
        var3 = Variable("var3", lambda x: np.ones(x))

        seg = (
            ((var1 > 3) & (var2 == 5)) |
            (
                (var1 == var3) &
                ((var2 < var1) | (var3 > 5))
            )
        )
        seg_variables = seg.variables
        breakpoint()

class TestSegmentationSegment:
    def test_general_working(self):
        var = Variable("var", lambda x: np.ones(x))
        seg = (var >= 3)
        data = pd.DataFrame({"var": [1, 2, 3, 4, 5]})
        segment_mask = seg.segment(data)
        pd.testing.assert_series_equal(
            segment_mask,
            pd.Series([False, False, True, True, True], name="var")
        )

    def test_multiple_segments(self):
        var1 = Variable("var1", lambda x: np.ones(x))
        var2 = Variable("var2", lambda x: np.ones(x))
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
        var1 = Variable("var1", lambda x: np.ones(x))
        var2 = Variable("var2", lambda x: np.ones(x))
        seg1 = var1 >= 4
        seg2 = var2 < 5
        seg3 = (var1 == 2)

        #breakpoint()
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

