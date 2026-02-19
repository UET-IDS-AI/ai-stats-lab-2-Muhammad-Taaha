"""
AI Mathematical Tools – Probability & Random Variables

Instructions:
- Implement ALL functions.
- Do NOT change function names or signatures.
- Do NOT print inside functions.
- You may use: math, numpy, matplotlib.
"""
import random
import math
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Part 1 — Probability Foundations
# ============================================================

def probability_union(PA, PB, PAB):
    """
    P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
    """
    information = PA + PB - PAB
    return information


def conditional_probability(PAB, PB):
    """
    P(A|B) = P(A ∩ B) / P(B)
    """
    information = PAB / PB
    return information


def are_independent(PA, PB, PAB, tol=1e-9):
    """
    True if:
        |P(A ∩ B) - P(A)P(B)| < tol
    """
    return abs(PAB - PA * PB) < tol


def bayes_rule(PBA, PA, PB):
    """
    P(A|B) = P(B|A)P(A) / P(B)
    """
    information = PBA*PA / PB
    return information


# ============================================================
# Part 2 — Bernoulli Distribution
# ============================================================

def bernoulli_pmf(x, theta):
    if x not in (0, 1):
        return 0
    if theta < 0 or theta > 1:
        return 0
    return (theta**x) * ((1-theta)**(1-x))


def bernoulli_theta_analysis(theta_values):
    """
    Returns:
        (theta, P0, P1, is_symmetric)
    """
    results = []
    for theta in theta_values:
        P0 = theta
        P1 = 1 - P0
        is_symmetric = False
        if P0 == P1:
            is_symmetric = True

    results.append((theta, P0, P1, is_symmetric))
    return results


# ============================================================
# Part 3 — Normal Distribution
# ============================================================

def normal_pdf(x, mu, sigma):
    """
    Normal PDF:
        1/(sqrt(2π)σ) * exp(-(x-μ)^2 / (2σ^2))
    """
    coefficient = 1 / (math.sqrt(2 * math.pi) * sigma)
    exponent = math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    return coefficient * exponent


def normal_histogram_analysis(mu_values,
                              sigma_values,
                              n_samples=10000,
                              bins=30):

    results = []

    length = len(mu_values)

    for i in range(length):

        mu = mu_values[i]
        sigma = sigma_values[i]

        samples = []
        for _ in range(n_samples):
            samples.append(random.gauss(mu, sigma))

        # Sample mean
        total = 0
        for value in samples:
            total += value
        sample_mean = total / n_samples

        # Sample variance
        total_var = 0
        for value in samples:
            total_var += (value - sample_mean) ** 2
        sample_variance = total_var / n_samples

        theoretical_mean = mu
        theoretical_variance = sigma ** 2

        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            mu,
            sigma,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results
# ============================================================
# Part 4 — Uniform Distribution
# ============================================================


def uniform_mean(a, b):
    """
    (a + b) / 2
    """
    return (a + b) / 2


def uniform_variance(a, b):
    """
    (b - a)^2 / 12
    """
    return ((b - a) ** 2) / 12


def uniform_histogram_analysis(a_values,
                               b_values,
                               n_samples=10000,
                               bins=30):

    results = []

    length = len(a_values)

    for i in range(length):

        a = a_values[i]
        b = b_values[i]

        # Generate samples
        samples = []
        for _ in range(n_samples):
            samples.append(random.uniform(a, b))

        # Sample mean
        total = 0
        for value in samples:
            total += value
        sample_mean = total / n_samples

        # Sample variance
        total_var = 0
        for value in samples:
            total_var += (value - sample_mean) ** 2
        sample_variance = total_var / n_samples

        # Theoretical values
        theoretical_mean = uniform_mean(a, b)
        theoretical_variance = uniform_variance(a, b)

        # Errors
        mean_error = abs(sample_mean - theoretical_mean)
        variance_error = abs(sample_variance - theoretical_variance)

        results.append((
            a,
            b,
            sample_mean,
            theoretical_mean,
            mean_error,
            sample_variance,
            theoretical_variance,
            variance_error
        ))

    return results


if __name__ == "__main__":
    print("Implement all required functions.")
