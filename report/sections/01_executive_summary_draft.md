# Executive summary (DRAFT - numbers filled automatically, edit wording)

**Aim.** Help a film distributor forecast box-office revenue *before release* for both standalone
and franchise films, using only information known in advance.

**Data.** 7,276 films with known revenue from The Movies Dataset (Kaggle),
1915-2017. After careful cleaning we kept
98% of films with reported revenue; missing budgets (27%) were
imputed with a regression model and flagged.

**Key findings.**
- Franchise films earn about 5.3x more than standalone
  films (median revenue).
- Budget is the strongest single driver: +1% budget is associated with about
  0.56% more revenue for original films
  and 0.39% for sequels.
- Track record of director, lead cast and earlier franchise films adds predictive power
  beyond budget (test R² on log revenue 0.40 with budget only vs
  0.59 with the full model).

**Model performance on unseen films.** 33% of forecasts fall within 2x of actual
revenue. The blockbuster classifier (revenue >= $100M, only 19% of films)
reaches AUC 0.91 and catches 83% of real blockbusters
(precision 56%) after handling class imbalance.

**Recommendation.** Use the forecast as a planning range, not a point estimate; combine it with
marketing and competition information that the model does not see.
