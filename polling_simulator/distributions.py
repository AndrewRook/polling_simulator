from scipy import stats


def convert_generic_scipy_distribution(distribution, *args, **kwargs):
    distro = distribution(*args, **kwargs)
    return distro.rvs


def truncated_gaussian_distribution(mean, sigma, lower_clip, upper_clip):
    a = (lower_clip - mean) / sigma
    b = (upper_clip - mean) / sigma
    return convert_generic_scipy_distribution(stats.truncnorm, a, b, loc=mean, scale=sigma)