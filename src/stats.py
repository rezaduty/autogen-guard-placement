"""
Small-sample statistics used by analyze.py. Exact where the counts are small.

verify_numbers.py deliberately does not import this module. It carries its own
implementations so a bug here cannot confirm itself.
"""
import math


def wilson(k, n, z=1.959963984540054):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_exact(a, b, c, d):
    """Two-sided Fisher exact test on [[a, b], [c, d]]. Sums every table with
    the same margins whose probability does not exceed the observed one."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2

    def logp(x):
        return _log_comb(r1, x) + _log_comb(r2, c1 - x) - _log_comb(n, c1)

    lo, hi = max(0, c1 - r2), min(r1, c1)
    obs = logp(a)
    total = 0.0
    for x in range(lo, hi + 1):
        lp = logp(x)
        if lp <= obs + 1e-9:
            total += math.exp(lp)
    return min(1.0, total)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test on the discordant counts of a paired
    comparison: b pairs positive only under condition one, c only under two."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.exp(_log_comb(n, i) - n * math.log(2)) for i in range(k + 1))
    return min(1.0, 2 * p)


def auroc(pos, neg):
    """Probability that a random positive outscores a random negative, ties half."""
    if not pos or not neg:
        return float("nan")
    s = 0.0
    for a in pos:
        for b in neg:
            s += 1.0 if a > b else 0.5 if a == b else 0.0
    return s / (len(pos) * len(neg))
