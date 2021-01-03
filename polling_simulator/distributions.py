from scipy import stats


def convert_generic_scipy_distribution(distribution, *args, **kwargs):
    """
    Take any of the standard ``scipy.stats`` distributions and convert it into the
    format that the ``Variable`` class demands for its ``data_generator`` attribute.

    Parameters
    ----------
    distribution: The distribution (e.g. ``scipy.stats.truncnorm``)
    args: positional arguments to pass to instantiate ``distribution``
    kwargs keyword arguments to pass to instantiate ``distribution``

    Returns
    -------
    A function ready for use as a ``Variable.data_generator``

    """
    distro = distribution(*args, **kwargs)
    return distro.rvs


def truncated_gaussian_distribution(mean, sigma, lower_clip, upper_clip):
    """
    Helper function to generate a truncated gaussian distribution.

    Parameters
    ----------
    mean: The mean of the distribution
    sigma: The standard deviation of the distribution
    lower_clip: The minimum allowable value
    upper_clip: The maximum allowable value

    Returns
    -------
    A function ready for use as a ``Variable.data_generator``

    """
    a = (lower_clip - mean) / sigma
    b = (upper_clip - mean) / sigma
    return convert_generic_scipy_distribution(stats.truncnorm, a, b, loc=mean, scale=sigma)