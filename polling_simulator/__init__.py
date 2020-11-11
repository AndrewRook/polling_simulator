import operator

from typing import Union

class Demographic:
    def __init__(
            self,
            n_voters: int,
            turnout_likelihood: float,
            response_likelihood: float,
            candidate_preference: int,
            demographic_segmentation
    ):
        pass


class _Base:
    def __eq__(self, other: Union[int, float, str, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.eq)

    def __ne__(self, other: Union[int, float, str, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.ne)

    def __ge__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.ge)

    def __gt__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.gt)

    def __le__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.le)

    def __lt__(self, other: Union[int, float, "Variable", "Segmentation"]):
        return Segmentation(self, other, operator.lt)

    def __and__(self, other: Union["Variable", "Segmentation"]):
        return Segmentation(self, other, operator.and_)

    def __or__(self, other: Union["Variable", "Segmentation"]):
        return Segmentation(self, other, operator.or_)


class Variable(_Base):
    def __init__(self, name: str):
        self.name = name


class Segmentation(_Base):
    def __init__(
            self,
            left: Union[int, float, str, Variable, "Segmentation"],
            right: Union[int, float, str, Variable, "Segmentation"],
            comparator
    ):
        self.left = left
        self.right = right
        self.comparator = comparator

    def segment(self, df):
        # If Segmentation, first evaluate Segmentation
        left = self.left if issubclass(self.left.__class__, Segmentation) == False else self.left.segment(df)
        right = self.right if issubclass(self.right.__class__, Segmentation) == False else self.right.segment(df)
        # If Variable, replace with dataframe column
        left = left if issubclass(left.__class__, Variable) == False else df[left.name]
        right = right if issubclass(right.__class__, Variable) == False else df[right.name]

        return self.comparator(left, right)
