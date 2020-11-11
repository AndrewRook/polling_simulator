import pandas as pd

from polling_simulator import Variable, Segmentation


class TestVariable:

    def test_instantiates_ok(self):
        var = Variable("woo")
        assert var.name == "woo"


class TestSegmentation:
    def test_general_working(self):
        var = Variable("var")
        #breakpoint()
        seg = (var >= 3)
        data = pd.DataFrame({"var": [1, 2, 3, 4, 5]})
        segment_mask = seg.segment(data)
        pd.testing.assert_series_equal(
            segment_mask,
            pd.Series([False, False, True, True, True], name="var")
        )

    