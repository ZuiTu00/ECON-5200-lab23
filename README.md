# ECON-5200-lab23
## M — Metrics & Main Results

**Sample:** 240 FOMC meeting minutes, 2000–2026. Tightening base rate 30%.

**Clustering (K=3, unsupervised).**

| Representation | Silhouette | Cluster-year pattern |
|---|---|---|
| TF-IDF (SVD-50) | 0.158 | **Separates by vocabulary epoch** (2000–08 / 2008–18 / 2017–26) |
| Sentence embeddings (384-d) | 0.197 | **Time-overlapping clusters** — captures communication posture, not vocabulary |

Adjusted Rand Index between the two partitions = **0.36**. The two
representations agree on broad structure but disagree on the middle
period, which TF-IDF slices by word-choice shifts (Bernanke QE vocabulary
vs Powell normalization vocabulary) while embeddings group by underlying
policy stance. This divergence is itself a finding: **TF-IDF measures
"what the Fed said"; embeddings measure "what the Fed meant."**

**Supervised prediction (expanding-window 5-fold CV).**

| Representation | Mean AUC | Std | Usable folds |
|---|---|---|---|
| TF-IDF (SVD-50) | 0.800 | 0.243 | 3 |
| Sentence-Embeddings (384-d) | 0.714 | 0.213 | 3 |

Paired t-test: t = 0.72, p = 0.55 → **difference not statistically
significant**. Both methods fail in the final fold (AUC ≈ 0.47, worse
than random) when predicting the 2022–2023 Powell tightening from
pre-2020 training data — consistent with a structural break in FOMC
communication after COVID-19.

**Caveats (disclosed as limitations, not hidden).**
- First two CV folds dropped because the early FOMC sample (pre-2004) contains no tightening episodes — a known failure mode of expanding-window CV for rare, clustered targets.
- The target variable is a synthetic regime label based on known tightening years, not actual Fed Funds rate changes. A production version would use FRED data and predict t+1 rate changes from t-dated text.
- `all-MiniLM-L6-v2` is a general-purpose encoder. A finance-domain encoder (e.g., FinBERT) could materially change the embeddings comparison.

## E — Evaluation / Extensions

**Directions for further work.**

1. Replace the curated LM subset with the full Loughran-McDonald Master
   Dictionary (~2,700 negative terms).
2. Swap `all-MiniLM-L6-v2` for `FinBERT` or `FLANG-BERT`; re-run AUC
   comparison.
3. Replace the synthetic tightening label with actual Fed Funds rate
   changes from FRED; use a proper t+1 forecasting target.
4. Run a formal Chow test or Bai-Perron test on the communication
   features to confirm the 2020 structural break suggested by the
   fold-5 prediction failure.
5. Use rolling-window CV with a 24-month test block instead of
   expanding-window 5-fold, to recover more usable folds.

## References

- Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks." *Journal of Finance*, 66(1): 35–65.
- Hansen, S., McMahon, M. & Prat, A. (2018). "Transparency and Deliberation within the FOMC: A Computational Linguistics Approach." *Quarterly Journal of Economics*, 133(2): 801–870.
- Blinder, A., Ehrmann, M., Fratzscher, M., De Haan, J. & Jansen, D.-J. (2008). "Central Bank Communication and Monetary Policy: A Survey of Theory and Evidence." *Journal of Economic Literature*, 46(4): 910–945.
- Tetlock, P. (2007). "Giving Content to Investor Sentiment: The Role of Media in the Stock Market." *Journal of Finance*, 62(3): 1139–1168.

## Repository Layout
