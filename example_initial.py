"""
example_initial.py
==========

Minimal usage example for ``multinull_jsd``.

This script shows how a user might use :class:`multinull_jsd.MultiNullJSDTest` to compare several candidate discrete
probability models against an observed histogram.

Scenario
--------
Imagine a survey question with three possible answers:

1. "Support"
2. "Neutral"
3. "Oppose"

You collect 100 responses and get the following counts:

- 55 people chose "Support"
- 22 people chose "Neutral"
- 23 people chose "Oppose"

You also have two candidate models (null hypotheses) for how people *should* respond, if your theory is correct:

- H1 (Baseline model):           [0.50, 0.30, 0.20]
- H2 (Alternative model):        [0.40, 0.40, 0.20]

Each null hypothesis comes with a desired significance level (alpha):

- H1: alpha = 0.05
- H2: alpha = 0.01

This script:

1. Builds a MultiNullJSDTest object.
2. Registers both null hypotheses with their target alpha.
3. Evaluates the observed histogram.
4. Prints the p-values for each null.
5. Prints which null is selected, or whether all are rejected in favor of the alternative (data do not fit any null).

How to run
----------
Assuming you have installed the package in your environment:

    python example_initial.py

or:

    python -m multinull_jsd.example
"""

from __future__ import annotations

from multinull_jsd import MultiNullJSDTest


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Observed data: histogram of counts
    # ------------------------------------------------------------------
    observed_counts: list[int] = [55, 22, 23]
    evidence_size: int = sum(observed_counts)  # Total number of observations
    prob_dim: int = len(observed_counts)       # Number of categories

    # In your own project, you would typically build `observed_counts` from your data using,
    # e.g., collections.Counter or pandas.value_counts.

    # ------------------------------------------------------------------
    # 2. Define candidate null hypotheses
    # ------------------------------------------------------------------
    # Each null is a list of probabilities that sum to 1.
    null_labels: list[str] = [
        "Baseline model: 50% support, 30% neutral, 20% oppose",
        "Alternative model: 40% support, 40% neutral, 20% oppose",
    ]

    null_probabilities: list[list[float]] = [
        [0.5, 0.3, 0.2],  # H1
        [0.4, 0.4, 0.2],  # H2
    ]

    # Target significance levels (Type I error rates) for each null
    target_alphas: list[float] = [0.05, 0.01]

    # ------------------------------------------------------------------
    # 3. Build the MultiNullJSDTest object
    # ------------------------------------------------------------------
    # - set evidence_size and prob_dim;
    # - use the "mc_multinomial" backend (recommended); alternatively, use "exact" for small n or "mc_normal";
    # - choose a reasonable number of Monte Carlo samples (e.g., 10_000);
    # - set a random seed for reproducibility.
    test = MultiNullJSDTest(
        evidence_size=evidence_size,
        prob_dim=prob_dim,
        cdf_method="mc_multinomial",
        mc_samples=10_000,
        seed=0,
    )

    # Register each null with its target alpha
    for probs, alpha in zip(null_probabilities, target_alphas, strict=True):
        test.add_nulls(prob_vector=probs, target_alpha=alpha)

    # ------------------------------------------------------------------
    # 4. Compute p-values and decisions for the observed histogram
    # ------------------------------------------------------------------
    # p_values: one p-value per null hypothesis (in the same order as added).
    p_values = test.infer_p_values(hist_query=observed_counts)  # FloatArray

    # decisions: an integer code per test. For a single histogram, this is a 1D array of length 1:
    #
    #   -1     → all null hypotheses are rejected (alternative hypothesis)
    #    k ≥ 1 → index (1-based) of the selected null hypothesis
    #
    # In this example we only have one histogram, so test.infer_decisions returns an integer scalar.
    decision_code = test.infer_decisions(hist_query=observed_counts)

    # ------------------------------------------------------------------
    # 5. Print results in a human-friendly way
    # ------------------------------------------------------------------
    print("=== MultiNullJSDTest example ===\n")

    print("Observed histogram (counts per category):")
    print(f"  Support, Neutral, Oppose = {observed_counts}")
    print(f"  Total responses (evidence_size): {evidence_size}\n")

    print("Candidate null hypotheses:")
    for idx, (label, probs, alpha, p_val) in enumerate(
        zip(null_labels, null_probabilities, target_alphas, p_values, strict=True),
        start=1,
    ):
        print(f"H{idx}: {label}")
        print(f"  Probabilities: {probs}")
        print(f"  Target alpha: {alpha:.3f}")
        print(f"  p-value:      {float(p_val):.4f}\n")

    # Interpret the decision code
    if decision_code == -1:
        print(
            "Decision:\n"
            "  The test rejected all candidate models at their requested\n"
            "  significance levels. The data are better explained by the\n"
            "  alternative hypothesis (i.e., none of the registered nulls\n"
            "  provides an adequate fit)."
        )
    else:
        selected_index: int = decision_code - 1  # 1-based → 0-based
        selected_label: str = null_labels[selected_index]
        print("Decision:")
        print(
            f"  The test selected null hypothesis H{decision_code},\n"
            f"  which corresponds to:\n"
            f"  {selected_label}"
        )

    print("\nNote:")
    print(
        "  - You can change the observed counts, candidate probabilities,\n"
        "    and target alphas to match your own study.\n"
        "  - For more advanced usage (batch processing, diagnostics, etc.),\n"
        "    see the package documentation and the docstrings of\n"
        "    'MultiNullJSDTest'."
    )


if __name__ == "__main__":
    main()
