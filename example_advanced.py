"""
example_advanced.py
===================

Advanced usage example for ``multinull_jsd``.

TL;DR
-----
Use this script if you want to do **more than a single, simple test**.

This example shows how to:
- Compare **several candidate models at once** (multiple null hypotheses).
- Evaluate **many datasets in a batch** instead of one-by-one.
- Check how **reliable** the procedure is in practice by simulation:
  - How often it makes a **false alarm** (Type I error / alpha).
  - How often it **fails to detect** a real change (Type II error / beta).
- See how different internal options (backends) like ``mc_multinomial`` vs ``exact`` behave on the same data.
- Get a rough sense of the **overall risk of false positives** if you plan to run the test many times
  (FWER approximation).

If you just want to plug in one observed histogram and get p-values and a decision, start with ``example_initial.py``
instead. This example is for *planning* and *evaluating* a study design:
“How often will this test be right or wrong under different scenarios?”

The focus here is *how to drive the library from user code* rather than the internal calibration logic of
:class:`multinull_jsd.MultiNullJSDTest`.

Scenario
--------
We consider a categorical outcome with three possible categories, e.g.:

1. "Support"
2. "Neutral"
3. "Oppose"

We work with total sample size n = 20 (20 individuals / observations).

We define three candidate null distributions over these categories:

- H1: [0.50, 0.30, 0.20]
- H2: [0.60, 0.25, 0.15]
- H3: [0.30, 0.30, 0.40]

and assign the same target alpha to each (0.05). The test is used to select one of these as "best supported" by the
data or to reject them all in favor of a catch-all alternative.

We then:

* Evaluate a *batch* of histograms under H1 using the Monte Carlo backend.
* Compare the p-values produced by ``mc_multinomial`` and ``exact`` for a single histogram.
* Use simulation + ``infer_decisions`` to estimate:

    alpha_ℓ ≈ P(decision ≠ ℓ | H_ℓ is true)
    beta_ℓ  ≈ P(decision = ℓ | data come from an alternative)

  and show how one might translate an estimated alpha_ℓ into a simple FWER approximation when applying the same test
  many times independently.

How to run
----------
Assuming you have installed the package and NumPy in your environment:

    python example_advanced.py
"""

from __future__ import annotations

import numpy as np

from multinull_jsd import MultiNullJSDTest


# ----------------------------------------------------------------------
# Helper functions for simulation-based operating characteristics
# ----------------------------------------------------------------------


def simulate_decisions(
    test: MultiNullJSDTest,
    generating_probs: list[float],
    evidence_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate `n_replications` histograms from a multinomial model with given probabilities and feed them to
    `test.infer_decisions`.

    Parameters
    ----------
    test
        A configured :class:`MultiNullJSDTest` instance.
    generating_probs
        Probabilities used to generate synthetic data (length = prob_dim).
    evidence_size
        Total count per histogram (must match the `evidence_size` used when building `test`).
    n_replications
        Number of Monte Carlo replications.
    rng
        NumPy random Generator for reproducibility.

    Returns
    -------
    np.ndarray
        1D array of decision codes (shape: (n_replications,)), where each entry is:

        -1           → all null hypotheses rejected (alternative chosen)
         k ≥ 1       → index (1-based) of selected null hypothesis
    """
    hist_batch = rng.multinomial(
        n=evidence_size,
        pvals=np.asarray(generating_probs, dtype=float),
        size=n_replications,
    )
    decisions = test.infer_decisions(hist_query=hist_batch)
    # Ensure we always work with a flat NumPy array of ints.
    return np.asarray(a=decisions, dtype=int).reshape(-1)


def estimate_alpha_for_null(
    test: MultiNullJSDTest,
    null_index: int,
    null_probabilities: list[list[float]],
    evidence_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> float:
    """
    Empirical estimate of α_{n,ℓ} = P(φ_n(H) = -1 | H_ℓ is true).

    Parameters
    ----------
    test
        A configured :class:`MultiNullJSDTest` instance.
    null_index
        Index (1-based) of the null hypothesis to evaluate.
    null_probabilities
        List of probability vectors for each null hypothesis.
    evidence_size
        Total count per histogram (must match the `evidence_size` used when building `test`).
    n_replications
        Number of Monte Carlo replications.
    rng
        NumPy random Generator for reproducibility.

    Returns
    -------
    float
        Empirical estimate of alpha_ℓ.
    """
    generating_probs = null_probabilities[null_index - 1]
    decisions = simulate_decisions(
        test=test,
        generating_probs=generating_probs,
        evidence_size=evidence_size,
        n_replications=n_replications,
        rng=rng,
    )

    alpha_hat = float(np.mean(decisions == -1))
    return alpha_hat


def estimate_beta(
    test: MultiNullJSDTest,
    alternative_probs: list[float],
    evidence_size: int,
    n_replications: int,
    rng: np.random.Generator,
) -> float:
    """
    Empirical estimate of β_n^q = P(φ_n(H) ∈ [L] | p = q_alt).

    Parameters
    ----------
    test
        A configured :class:`MultiNullJSDTest` instance.
    alternative_probs
        Probability vector for the alternative distribution.
    evidence_size
        Total count per histogram (must match the `evidence_size` used when building `test`).
    n_replications
        Number of Monte Carlo replications.
    rng
        NumPy random Generator for reproducibility.

    Returns
    -------
    float
        Empirical estimate of beta.
    """
    decisions = simulate_decisions(
        test=test,
        generating_probs=alternative_probs,
        evidence_size=evidence_size,
        n_replications=n_replications,
        rng=rng,
    )

    beta_hat = float(np.mean(decisions != -1))
    return beta_hat


# ----------------------------------------------------------------------
# Main example
# ----------------------------------------------------------------------


def main() -> None:
    # Reproducible global RNG for this example
    rng: np.random.Generator = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # 1. Define null hypotheses and basic dimensions
    # ------------------------------------------------------------------
    categories: list[str] = ["Support", "Neutral", "Oppose"]
    prob_dim: int = len(categories)

    evidence_size: int = 20  # total count per histogram

    null_labels: list[str] = [
        "H1: Baseline (50% support, 30% neutral, 20% oppose)",
        "H2: More support (60% support, 25% neutral, 15% oppose)",
        "H3: More opposition (30% support, 30% neutral, 40% oppose)",
    ]

    null_probabilities: list[list[float]] = [
        [0.50, 0.30, 0.20],  # H1
        [0.60, 0.25, 0.15],  # H2
        [0.30, 0.30, 0.40],  # H3
    ]

    target_alphas: list[float] = [0.05, 0.05, 0.05]

    # Alternative distribution for beta estimation
    alternative_probs: list[float] = [0.20, 0.50, 0.30]

    # ------------------------------------------------------------------
    # 2. Build MultiNullJSDTest instances with different backends
    # ------------------------------------------------------------------
    # Monte Carlo backend (recommended for larger sample sizes)
    test_mc = MultiNullJSDTest(
        evidence_size=evidence_size,
        prob_dim=prob_dim,
        cdf_method="mc_multinomial",
        mc_samples=20_000,
        seed=0,
    )

    # Exact backend (feasible here because n=20 and prob_dim=3)
    test_exact = MultiNullJSDTest(
        evidence_size=evidence_size,
        prob_dim=prob_dim,
        cdf_method="exact",
    )

    # Normal (CLT) backend
    test_normal = MultiNullJSDTest(
        evidence_size=evidence_size,
        prob_dim=prob_dim,
        cdf_method="mc_normal",
        mc_samples=20_000,
        seed=0,
    )

    for probs, alpha in zip(null_probabilities, target_alphas, strict=True):
        test_normal.add_nulls(prob_vector=probs, target_alpha=alpha)
        test_mc.add_nulls(prob_vector=probs, target_alpha=alpha)
        test_exact.add_nulls(prob_vector=probs, target_alpha=alpha)

    # ------------------------------------------------------------------
    # 3. Batched evaluation under H1 using the Monte Carlo backend
    # ------------------------------------------------------------------
    print("=== MultiNullJSDTest example: batched evaluation & operating characteristics ===\n")

    batch_size: int = 8
    print(f"Generating a batch of {batch_size} histograms under H1...")
    hist_batch = rng.multinomial(
        n=evidence_size,
        pvals=np.asarray(null_probabilities[0], dtype=float),
        size=batch_size,
    )

    p_values_batch = test_mc.infer_p_values(hist_query=hist_batch)
    decisions_batch = np.asarray(test_mc.infer_decisions(hist_query=hist_batch), dtype=int).reshape(-1)

    print("\nBatched results (Monte Carlo backend):")
    print("Row | Histogram [Support, Neutral, Oppose] | Decision | p-values per null [H1, H2, H3]")
    print("----+--------------------------------------+----------+-------------------------------")
    for idx in range(batch_size):
        hist = hist_batch[idx]
        decision_code = decisions_batch[idx]
        p_row = np.asarray(p_values_batch[idx], dtype=float)
        hist_str = f"[{hist[0]:2d}, {hist[1]:2d}, {hist[2]:2d}]"
        p_str = ", ".join(f"{p_val:0.4f}" for p_val in p_row)
        print(f"{idx:3d} | {hist_str:36s} | {decision_code:8d} | [{p_str}]")

    # ------------------------------------------------------------------
    # 4. Compare backends on a single histogram
    # ------------------------------------------------------------------
    test_hist = hist_batch[0]
    print("\nComparing backends for a single histogram:")
    print(f"  Histogram: {test_hist.tolist()}")

    pc_normal = np.asarray(test_normal.infer_p_values(hist_query=test_hist), dtype=float).reshape(-1)
    p_mc = np.asarray(test_mc.infer_p_values(hist_query=test_hist), dtype=float).reshape(-1)
    p_exact = np.asarray(test_exact.infer_p_values(hist_query=test_hist), dtype=float).reshape(-1)

    print("  p-values (mc_normal)     :", ", ".join(f"{v:0.6f}" for v in pc_normal))
    print("  p-values (mc_multinomial):", ", ".join(f"{v:0.6f}" for v in p_mc))
    print("  p-values (exact)         :", ", ".join(f"{v:0.6f}" for v in p_exact))

    # ------------------------------------------------------------------
    # 5. Empirical operating characteristics (alpha, beta, FWER)
    # ------------------------------------------------------------------
    n_replications: int = 5_000  # Increase for more precision (at higher CPU cost)

    print("\nEstimating operating characteristics via simulation "
          f"(n_replications = {n_replications})...")
    print()

    alpha_hats: list[float] = []

    for ell in range(1, len(null_probabilities) + 1):
        alpha_theoretical = float(test_exact.get_alpha(null_index=ell))
        alpha_hat = estimate_alpha_for_null(
            test=test_mc,
            null_index=ell,
            null_probabilities=null_probabilities,
            evidence_size=evidence_size,
            n_replications=n_replications,
            rng=rng,
        )

        alpha_hats.append(alpha_hat)

        print(f"For {null_labels[ell - 1]}")
        print(f"  Target alpha:  {target_alphas[ell - 1]:0.4f}")
        print(f"  Theoretical alpha (Type I): {alpha_theoretical:0.4f}")
        print(f"  Empirical alpha (Type I): {alpha_hat:0.4f}")
        print()

    beta_theoretical = float(test_exact.get_beta(prob_query=alternative_probs))
    beta_hat = estimate_beta(
        test=test_mc,
        alternative_probs=alternative_probs,
        evidence_size=evidence_size,
        n_replications=n_replications,
        rng=rng,
    )

    print(f"For an alternative: {alternative_probs}")
    print(f"  Theoretical beta (Type II vs alt={alternative_probs}): {beta_theoretical:0.4f}")
    print(f"  Empirical beta  (Type II vs alt={alternative_probs}): {beta_hat:0.4f}")
    print()

    alpha_star = test_exact.get_fwer()
    print(f"Family-wise error rate (FWER) of the test: {alpha_star:0.4f}")

    print(
        "\nNotes:\n"
        "  - The alpha/beta estimates above are Monte Carlo approximations based on\n"
        "    simulation. Increase `n_replications` for more stable estimates.\n"
        "  - The simple FWER calculation assumes independent applications of the\n"
        "    same test. In complex multiple-testing settings, dependency between\n"
        "    tests must be taken into account.\n"
        "  - You can adapt the nulls, alternative distribution, backends, and\n"
        "    sample size to match your own study design."
    )


if __name__ == "__main__":
    main()
