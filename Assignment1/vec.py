import math


class Vec:
    """A basic Vector class storing numeric elements in a tuple."""

    def __init__(self, data=None):
        if data is None:
            self.elements = ()
        else:
            self.elements = tuple(data)

    def __repr__(self):
        return f"Vec{self.elements}"

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, index):
        return self.elements[index]

    def scalar_mult(self, c):
        """Multiply every element by a scalar value c."""
        return Vec(c * x for x in self.elements)

    def mean(self):
        """Compute the arithmetic mean of the vector elements."""
        n = len(self.elements)
        if n == 0:
            raise ValueError("Mean is undefined for an empty vector.")
        return sum(self.elements) / n

    def demean(self):
        """Subtract the mean from every element, centering the vector at zero."""
        mu = self.mean()
        return Vec(x - mu for x in self.elements)

    def std(self):
        """Compute the population standard deviation using demean()."""
        n = len(self.elements)
        if n == 0:
            raise ValueError("Standard deviation is undefined for an empty vector.")
        centered = self.demean()
        variance = sum(x ** 2 for x in centered.elements) / n
        return math.sqrt(variance)


if __name__ == "__main__":
    v = Vec([1, 2, 3, 4, 5])
    print("Vector:", v)
    print("Mean:", v.mean())
    print("Demeaned:", v.demean())
    print("Std Dev:", v.std())
