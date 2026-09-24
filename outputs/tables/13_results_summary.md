# Results summary (auto-generated)


## Data flow

| step | action | rows_before | rows_after | removed | reason |
|---|---|---|---|---|---|
| 01 load & dedupe metadata | drop | 45,466 | 45,433 | 33 | 3 malformed (shifted-column) rows + duplicate ids removed |
| 02a keep movies with known revenue | filter | 45,433 | 7,398 | 38,035 | revenue = 0 means unknown |
| 02b status conflict | drop | 7,398 | 7,385 | 13 | Rumored/Post Production with revenue |
| 02c missing release date | drop | 7,385 | 7,385 | 0 | no date |
| 02d fix unit errors (x1e6) | fix | 7,385 | 7,385 | 0 | budget & revenue both recorded in millions |
| 02e unreliable revenue < 1000 | drop | 7,385 | 7,276 | 109 | cannot verify unit |


## Data errors and handling

| check | n_rows | action | note |
|---|---|---|---|
| Malformed rows (shifted columns, non-numeric id) | 3 | removed | company names leaked into 'genres' |
| Exact duplicate rows in metadata | 17 | removed |  |
| Duplicate ids in metadata | 30 | removed (kept first) | differ only in popularity/vote_count |
| Duplicate ids in credits | 44 | removed (kept most complete crew) |  |
| Duplicate ids in keywords | 987 | removed | fully identical rows |
| Invalid genre labels after cleaning | 0 | checked | none |
| Revenue = 0 (unknown box office) | 38,035 | excluded | target variable cannot be imputed |
| Revenue reported but status not 'Released' | 13 | removed | {'Rumored': 6, 'Post Production': 5} |
| Missing release date | 0 | removed | needed for time-aware features |
| Budget AND revenue < 1000 (recorded in millions) | 43 | fixed x 1,000,000 (kept) |  |
| Revenue < 1000 not fixable | 109 | removed | unreliable target |
| Budget < 1000 with valid revenue | 9 | set to missing -> imputed (kept) |  |
| Budget = 0 (unknown) | 1,921 | set to missing -> imputed (kept) |  |
| Runtime = 0 (unknown) | 17 | set to missing -> imputed (kept) |  |
| Runtime > 300 min | 2 | flagged (kept) |  |
| Extreme ROI (>1000x or <0.001x) | 15 | flagged (kept) | e.g. micro-budget hits; log scale reduces influence |
| Same title + same year, different id | 0 | checked (kept) | different films sharing a title |
| Director gender code 0/missing | 1,418 | recoded 'unknown' |  |
| Lead actor gender code 0/missing | 581 | recoded 'unknown' |  |
| Post-release columns (popularity, votes) | 6 | removed (leakage) | POST_popularity, POST_vote_average, POST_vote_count |
| belongs_to_collection is retrospective | 1,471 | kept for EDA only | model uses is_sequel from agent 06 |


## Imputation plan

| variable | n_imputed | method |
|---|---|---|
| genres | 18 | meaningful value 'Unknown' |
| production_companies | 375 | meaningful value 'Unknown' |
| production_countries | 128 | meaningful value 'Unknown' |
| spoken_languages | 49 | meaningful value 'Unknown' |
| directors | 9 | meaningful value 'Unknown' |
| top5_cast | 27 | meaningful value 'Unknown' |
| writers | 281 | meaningful value 'Unknown' |
| keywords | 627 | empty = none |
| composer | 3,054 | meaningful value 'Unknown' |
| collection_name | 5,805 | empty = none |
| runtime | 22 | median runtime of the same primary genre |
| cast_female_share | 112 | median + flag cast_gender_unknown |
| budget | 1,930 | model-based on log(budget) + flag budget_missing |


## Budget imputation - 5-fold CV

| method | cv5_r2_mean | cv5_r2_std | used |
|---|---|---|---|
| linear regression on log(budget) | 0.504 | 0.036 | 1 |
| median by primary genre x decade | 0.215 | 0.017 | 0 |


## Descriptive statistics (continuous)

| variable | n | mean | std | min | q1 | median | q3 | max | iqr | cv | skew | kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| revenue | 7,276 | 70,516,005.147 | 147,919,745.916 | 1,081.000 | 2,998,952.750 | 17,725,226.500 | 69,625,564.250 | 2,787,965,087.000 | 66,626,611.500 | 2.098 | 5.078 | 41.838 |
| budget | 7,276 | 25,867,731.686 | 38,914,697.268 | 2,500.000 | 4,500,000.000 | 12,000,000.000 | 30,000,000.000 | 592,000,000.000 | 25,500,000.000 | 1.504 | 3.902 | 26.273 |
| runtime | 7,276 | 108.174 | 20.680 | 25.000 | 94.000 | 104.000 | 118.000 | 338.000 | 24.000 | 0.191 | 1.641 | 6.930 |
| year | 7,276 | 1,999.665 | 15.415 | 1,915.000 | 1,992.000 | 2,004.000 | 2,011.000 | 2,017.000 | 19.000 | 0.008 | -1.708 | 3.859 |
| cast_size | 7,276 | 21.576 | 18.947 | 0.000 | 11.000 | 16.000 | 25.000 | 313.000 | 14.000 | 0.878 | 3.650 | 23.863 |
| crew_size | 7,276 | 25.167 | 30.291 | 0.000 | 8.000 | 15.000 | 29.000 | 435.000 | 21.000 | 1.204 | 2.944 | 14.143 |
| n_keywords | 7,276 | 7.314 | 6.264 | 0.000 | 3.000 | 6.000 | 10.000 | 149.000 | 7.000 | 0.857 | 3.139 | 43.608 |
| n_companies | 7,276 | 2.764 | 2.185 | 0.000 | 1.000 | 2.000 | 4.000 | 26.000 | 3.000 | 0.791 | 2.328 | 11.082 |
| n_countries | 7,276 | 1.338 | 0.789 | 0.000 | 1.000 | 1.000 | 1.000 | 12.000 | 0.000 | 0.589 | 3.036 | 16.466 |
| n_genres | 7,276 | 2.503 | 1.118 | 0.000 | 2.000 | 2.000 | 3.000 | 8.000 | 1.000 | 0.447 | 0.471 | -0.065 |
| n_spoken_languages | 7,276 | 1.450 | 0.892 | 0.000 | 1.000 | 1.000 | 2.000 | 9.000 | 1.000 | 0.615 | 2.537 | 8.697 |
| n_writers | 7,276 | 2.235 | 1.649 | 0.000 | 1.000 | 2.000 | 3.000 | 23.000 | 2.000 | 0.738 | 2.861 | 17.623 |
| n_producers | 7,276 | 3.189 | 3.029 | 0.000 | 1.000 | 3.000 | 5.000 | 26.000 | 4.000 | 0.950 | 1.620 | 4.266 |
| cast_female_share | 7,276 | 0.329 | 0.182 | 0.000 | 0.200 | 0.320 | 0.444 | 1.000 | 0.244 | 0.553 | 0.551 | 0.724 |
| budget (observed only) | 5,346 | 32,078,570.242 | 43,457,672.789 | 2,500.000 | 5,815,000.000 | 17,000,000.000 | 40,000,000.000 | 592,000,000.000 | 34,185,000.000 | 1.355 | 3.411 | 20.528 |
| log_revenue | 7,276 | 16.235 | 2.527 | 6.986 | 14.914 | 16.690 | 18.059 | 21.749 | 3.145 | 0.156 | -0.851 | 0.475 |
| log_budget | 7,276 | 16.183 | 1.510 | 7.824 | 15.320 | 16.300 | 17.217 | 20.199 | 1.897 | 0.093 | -0.722 | 1.394 |


## Categorical modes

| variable | n_categories | mode | mode_count | mode_proportion |
|---|---|---|---|---|
| primary_genre | 21 | Drama | 1,898 | 0.261 |
| original_language | 44 | en | 6,258 | 0.860 |
| month | 12 | 9.0 | 891 | 0.122 |
| weekday | 7 | Friday | 3,194 | 0.439 |
| decade | 11 | 2010 | 2,249 | 0.309 |
| segment | 2 | Standalone | 5,805 | 0.798 |
| director_gender_cat | 3 | male | 5,529 | 0.760 |
| lead_gender_cat | 3 | male | 5,014 | 0.689 |
| budget_missing | 2 | 0 | 5,346 | 0.735 |
| runtime_imputed | 2 | 0 | 7,254 | 0.997 |
| genres (multi-label) | 21 | Drama | 3,611 | 0.496 |


## Segment summary

| franchise_segment | n | median_revenue | mean_revenue | median_budget | blockbuster_rate | share |
|---|---|---|---|---|---|---|
| Standalone | 5,805 | 12,884,923.000 | 46,130,583.743 | 11,000,000.000 | 0.136 | 0.798 |
| Franchise starter | 680 | 65,577,274.000 | 144,747,299.144 | 15,000,000.000 | 0.375 | 0.093 |
| Sequel | 791 | 70,992,898.000 | 185,661,506.196 | 23,000,000.000 | 0.430 | 0.109 |


## Class imbalance

| variable | positives | negatives | positive_share | imbalance_ratio_neg_to_pos |
|---|---|---|---|---|
| Blockbuster (revenue >= threshold) | 1,384 | 5,892 | 0.190 | 4.257 |
| Sequel | 791 | 6,485 | 0.109 | 8.198 |
| Belongs to a franchise | 1,471 | 5,805 | 0.202 | 3.946 |


## Top 10 features by mutual information

| feature | label | mi_log_revenue | mi_blockbuster |
|---|---|---|---|
| log_budget | Production budget | 0.358 | 0.188 |
| lead_company | Lead production company | 0.342 | 0.171 |
| director_prior_mean_log_rev | Director track record (past revenue) | 0.154 | 0.078 |
| log_crew_size | Crew size | 0.128 | 0.070 |
| log_cast_size | Cast size | 0.123 | 0.056 |
| budget_missing | Budget not reported | 0.111 | 0.050 |
| cast_prior_max_log_rev | Biggest past hit of lead cast | 0.105 | 0.051 |
| collection_prior_mean_log_rev | Earlier franchise films' revenue | 0.104 | 0.053 |
| cast_prior_mean_log_rev | Lead cast track record | 0.087 | 0.039 |
| n_keywords | Number of keywords | 0.071 | 0.036 |


## Model comparison

| model | n_params | AIC_log_scale | BIC_log_scale | AIC_revenue_scale | BIC_revenue_scale | train_R2_adj | smearing | test_RMSE_log | test_MAE_log | test_R2_log | test_RMSE_usd | test_MAE_usd | test_MedianAE_usd | test_MedianAPE_pct | test_within_2x_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_OLS_budget_only | 2 | 24,982.802 | 24,996.140 | 53,140.854 | 53,154.192 | 0.333 | 5.214 | 1.935 | 1.446 | 0.401 | 155,878,550.048 | 84,512,995.680 | 43,254,042.861 | 235.947 | 27.610 |
| M2_OLS_full | 83 | 22,715.717 | 23,269.249 | 50,873.769 | 51,427.301 | 0.555 | 3.123 | 1.602 | 1.204 | 0.589 | 424,908,739.438 | 123,790,600.336 | 21,625,969.632 | 156.971 | 32.280 |
| M3_OLS_interaction | 84 | 22,711.742 | 23,271.943 | 50,869.795 | 51,429.995 | 0.555 | 3.138 | 1.602 | 1.203 | 0.589 | 386,150,941.941 | 119,768,185.863 | 21,644,762.614 | 156.860 | 32.074 |
| M4_OLS_selected | 51 | 22,680.764 | 23,020.886 | 50,838.817 | 51,178.938 | 0.555 | 3.165 | 1.601 | 1.204 | 0.590 | 383,777,986.114 | 120,025,754.486 | 21,150,575.921 | 153.906 | 32.555 |
| M5_GLM_Gamma_log | 51 |  |  | 51,048.841 | 51,388.963 | 0.476 | 1.000 | 1.911 | 1.334 | 0.415 | 109,607,579.081 | 45,892,246.645 | 15,613,307.578 | 84.809 | 41.346 |


## Selected OLS - 15 strongest significant terms

| label | coef | p_value | p_value_HC3 | effect_pct | unit |
|---|---|---|---|---|---|
| Production budget | 0.840 | 0.000 | 0.000 | 131.548 | per +1 SD |
| Budget not reported | -1.263 | 0.000 | 0.000 | -71.720 | 0 -> 1 |
| Lead production company: Other | -0.756 | 0.000 | 0.000 | -53.039 | vs reference category |
| Release year | -0.313 | 0.000 | 0.000 | -26.860 | per +1 SD |
| Director track record (past revenue) | 0.259 | 0.000 | 0.000 | 29.612 | per +1 SD |
| Lead production company: Unknown | -1.285 | 0.000 | 0.000 | -72.335 | vs reference category |
| Genre: Drama | -0.632 | 0.000 | 0.000 | -46.843 | 0 -> 1 |
| Number of keywords | 0.178 | 0.000 | 0.000 | 19.536 | per +1 SD |
| Number of genres | 0.429 | 0.000 | 0.000 | 53.569 | per +1 SD |
| Director has track record | 0.442 | 0.000 | 0.000 | 55.605 | 0 -> 1 |
| Genre: Science Fiction | -0.631 | 0.000 | 0.000 | -46.770 | 0 -> 1 |
| Runtime | 0.189 | 0.000 | 0.000 | 20.765 | per +1 SD |
| Genre: Thriller | -0.504 | 0.000 | 0.000 | -39.586 | 0 -> 1 |
| Original language: hi | 1.378 | 0.000 | 0.000 | 296.617 | vs reference category |
| Cast size | 0.169 | 0.000 | 0.000 | 18.409 | per +1 SD |


## Sensitivity (observed budgets only)

| model | n_train | coef_log_budget | coef_is_sequel | train_R2_adj | test_RMSE_log_observed_budget_films |
|---|---|---|---|---|---|
| Selected OLS, all films (budget imputed where missing) | 5,820 | 0.840 | -0.246 | 0.555 | 1.459 |
| Same features, observed-budget films only | 4,272 | 0.912 | -0.598 | 0.541 | 1.426 |


## Assumption tests

| assumption | test | statistic | p_value | conclusion | mitigation |
|---|---|---|---|---|---|
| 1. Independence | Durbin-Watson (ordered by release date) | 1.950 |  | OK (1.5-2.5) | films are separate products; time trend captured by year |
| 2. Normality of residuals | Kolmogorov-Smirnov vs N(0,1) | 0.073 | 0.000 | deviation from normal (check Q-Q) | log transform; large n -> CLT makes coefficient inference robust |
| 2. Normality of residuals | Jarque-Bera (skew -0.86, kurtosis 4.89) | 1,576.179 | 0.000 | heavy tails / skew | compare with GLM Gamma (M5) |
| 3. Linearity | Rainbow test | 0.922 | 0.985 | linear | log budget, interaction term; see observed vs predicted plot |
| 4. Constant variance | Breusch-Pagan | 580.639 | 0.000 | heteroscedastic | robust HC3 errors: 49 significant terms (OLS) vs 46 (HC3) |
| Multicollinearity | max VIF in selected model | 13.237 |  | high collinearity | backward elimination removed redundant predictors |


## Cross-validation (regression)

| model | RMSE_log_mean | RMSE_log_std | MAE_log_mean | MAE_log_std | R2_log_mean | R2_log_std | RMSE_usd_mean | RMSE_usd_std | MAE_usd_mean | MAE_usd_std | MedianAE_usd_mean | MedianAE_usd_std | MedianAPE_pct_mean | MedianAPE_pct_std | within_2x_pct_mean | within_2x_pct_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M1_OLS_budget_only | 2.069 | 0.071 | 1.524 | 0.034 | 0.333 | 0.013 | 157,274,101.737 | 3,906,685.318 | 87,177,514.907 | 3,599,147.481 | 44,785,844.938 | 2,808,374.974 | 229.744 | 19.498 | 29.021 | 0.971 |
| M2_OLS_full | 1,748,729,864.368 | 3,910,278,847.148 | 51,256,181.309 | 114,612,302.788 | -2,340,233,250,244,924,416.000 | 5,232,920,630,752,927,744.000 | 482,506,397.442 | 71,338,076.105 | 130,998,681.858 | 13,691,007.185 | 22,635,054.029 | 2,317,158.156 | 153.622 | 10.290 | 31.615 | 1.616 |
| M3_OLS_interaction | 1,723,633,725.630 | 3,854,162,174.959 | 50,520,600.448 | 112,967,493.997 | -2,273,545,542,078,073,600.000 | 5,083,802,382,028,180,480.000 | 434,516,620.937 | 45,743,849.780 | 126,586,939.133 | 11,326,196.056 | 23,015,878.073 | 2,506,623.125 | 156.640 | 10.754 | 31.770 | 1.411 |
| M4_OLS_selected | 1.707 | 0.056 | 1.265 | 0.033 | 0.545 | 0.022 | 416,609,329.557 | 30,158,636.414 | 124,504,552.044 | 10,963,836.251 | 23,365,846.002 | 2,009,823.981 | 159.272 | 11.571 | 31.718 | 1.323 |
| M5_GLM_Gamma_log | 2.035 | 0.083 | 1.391 | 0.038 | 0.354 | 0.021 | 98,887,017.675 | 13,458,148.482 | 45,264,791.675 | 3,652,732.535 | 16,842,019.633 | 815,386.354 | 83.051 | 3.370 | 41.340 | 0.885 |


## Cross-validation (classification)

| model | roc_auc_mean | average_precision_mean | recall_mean | precision_mean | roc_auc_std | average_precision_std | recall_std | precision_std |
|---|---|---|---|---|---|---|---|---|
| Logistic (unweighted) | 0.921 | 0.779 | 0.610 | 0.753 | 0.005 | 0.006 | 0.025 | 0.027 |
| Logistic (class_weight=balanced) | 0.920 | 0.776 | 0.846 | 0.545 | 0.005 | 0.006 | 0.022 | 0.014 |


## Classification metrics (test)

| model | threshold_rule | cv_expected_cost | test_expected_cost | threshold | AUC | AP | TP | FP | TN | FN | sensitivity_recall | specificity | precision | F1 | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Logistic (unweighted) | default 0.5 | 0.409 | 0.411 | 0.500 | 0.910 | 0.762 | 169 | 59 | 1,120 | 108 | 0.610 | 0.950 | 0.741 | 0.669 | 0.885 |
| Logistic (unweighted) | loss-matrix formula | 0.286 | 0.278 | 0.167 | 0.910 | 0.762 | 238 | 209 | 970 | 39 | 0.859 | 0.823 | 0.532 | 0.657 | 0.830 |
| Logistic (unweighted) | CV cost-optimal | 0.280 | 0.289 | 0.200 | 0.910 | 0.762 | 229 | 181 | 998 | 48 | 0.827 | 0.847 | 0.558 | 0.667 | 0.843 |
| Logistic (class_weight=balanced) | default 0.5 | 0.282 | 0.282 | 0.500 | 0.910 | 0.762 | 233 | 190 | 989 | 44 | 0.841 | 0.839 | 0.551 | 0.666 | 0.839 |
| Logistic (class_weight=balanced) | loss-matrix formula | 0.354 | 0.364 | 0.167 | 0.910 | 0.762 | 258 | 435 | 744 | 19 | 0.931 | 0.631 | 0.372 | 0.532 | 0.688 |
| Logistic (class_weight=balanced) | CV cost-optimal | 0.280 | 0.278 | 0.510 | 0.910 | 0.762 | 233 | 184 | 995 | 44 | 0.841 | 0.844 | 0.559 | 0.671 | 0.843 |


## Accuracy by segment (test)

| segment | n | RMSE_log | MAE_log | R2_log | RMSE_usd | MAE_usd | MedianAE_usd | MedianAPE_pct | within_2x_pct |
|---|---|---|---|---|---|---|---|---|---|
| All films | 1,456 | 1.601 | 1.204 | 0.590 | 383,777,986.114 | 120,025,754.486 | 21,150,575.921 | 153.906 | 32.555 |
| Standalone | 1,165 | 1.621 | 1.221 | 0.564 | 184,249,327.451 | 72,647,508.406 | 17,363,229.317 | 179.685 | 30.386 |
| Franchise starter | 133 | 1.735 | 1.331 | 0.390 | 208,968,459.197 | 95,026,877.489 | 31,604,276.016 | 66.217 | 54.135 |
| Sequel | 158 | 1.307 | 0.972 | 0.585 | 1,034,502,039.826 | 490,408,712.231 | 109,947,296.433 | 186.702 | 30.380 |


## Budget elasticity

{'standalone_or_starter': 0.5644691903956984, 'sequel': 0.39152917073105054}
