import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_csv  = "/users/davisturner/desktop/PriceCSVs/1319price.csv"
output_file = "/users/davisturner/desktop/PriceCSVs/maize_variability.csv"
plots_output = "/users/davisturner/desktop/PriceCSVs/"
panel = pd.read_csv(input_csv, low_memory=False)

# --- Remove flagged outliers ---
panel = panel[panel["outlier"] == 0].copy()

# --- Only keep households appearing in more than one round ---
round_counts = panel.groupby("household_id")["round"].count()
multi_round = round_counts[round_counts > 1].index
panel = panel[panel["household_id"].isin(multi_round)].copy()

# --- Compute within-household variability measures ---
hh_stats = panel.groupby("household_id").agg(
    price_cv        = ("price_per_kg", lambda x: x.std() / x.mean()),
    price_sd        = ("price_per_kg", "std"),
    price_range     = ("price_per_kg", lambda x: x.max() - x.min()),
    price_mean      = ("price_per_kg", "mean"),
    price_min       = ("price_per_kg", "min"),
    price_max       = ("price_per_kg", "max"),
    rounds_count    = ("round", "count"),
    sampling_weight = ("sampling_weight", "last")
).reset_index()

# --- Weighted summaries ---
weights      = hh_stats["sampling_weight"]
cv_values    = hh_stats["price_cv"]
sd_values    = hh_stats["price_sd"]
range_values = hh_stats["price_range"]

weighted_mean_cv    = np.average(cv_values, weights=weights)
weighted_mean_range = np.average(range_values, weights=weights)
weighted_mean_sd    = np.average(sd_values, weights=weights)

# --- Weighted summary table by round ---
print("\n=== Weighted Summary Statistics by Round ===")
round_summary = panel.groupby("round").apply(
    lambda x: pd.Series({
        "weighted_mean_price": np.average(x["price_per_kg"], weights=x["sampling_weight"]),
        "unweighted_mean_price": x["price_per_kg"].mean(),
        "std": x["price_per_kg"].std(),
        "min": x["price_per_kg"].min(),
        "max": x["price_per_kg"].max(),
        "n_households": len(x)
    })
).reset_index()

# --- Within-round CV (cross-sectional price dispersion per round) ---
# --- I added this on because I was unsure if this was the measure of price variability needed #
print("\n=== Within-Round CV (Cross-Sectional Price Dispersion) ===")
within_round_cv = panel.groupby("round").apply(
    lambda x: pd.Series({
        "within_round_cv": x["price_per_kg"].std() / x["price_per_kg"].mean(),
        "weighted_within_round_cv": (
            np.sqrt(np.average(
                (x["price_per_kg"] - np.average(x["price_per_kg"], weights=x["sampling_weight"]))**2,
                weights=x["sampling_weight"]
            ))
            / np.average(x["price_per_kg"], weights=x["sampling_weight"])
        ),
        "n_households": len(x)
    })
).reset_index()

# --- Plot 1: Boxplot of price_per_kg by round ---
fig, ax = plt.subplots(figsize=(10, 6))
round_groups = [panel[panel["round"] == r]["price_per_kg"].values
                for r in sorted(panel["round"].unique())]
ax.boxplot(round_groups, labels=sorted(panel["round"].unique()))
ax.set_xlabel("Survey Round")
ax.set_ylabel("Price per kg (ETB)")
ax.set_title("Distribution of Maize Price per kg by Survey Round")
plt.tight_layout()
plt.savefig(plots_output + "boxplot_by_round.png", dpi=150)
plt.close()

# --- Plot 2: Histogram of household-level CV ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(cv_values, bins=40, edgecolor="black", color="steelblue")
ax.axvline(weighted_mean_cv, color="red", linestyle="--",
           label=f"Weighted mean CV: {weighted_mean_cv:.3f}")
ax.set_xlabel("Coefficient of Variation")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Within-Household Maize Price Variability (CV)")
ax.legend()
plt.tight_layout()
plt.savefig(plots_output + "histogram_cv.png", dpi=150)
plt.close()


# --- Plot 3: Histogram of household-level SD ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sd_values, bins=40, edgecolor="black", color="steelblue")
ax.axvline(weighted_mean_sd, color="red", linestyle="--",
           label=f"Weighted mean SD: {weighted_mean_sd:.2f} ETB/kg")
ax.set_xlabel("Standard Deviation (ETB/kg)")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Within-Household Maize Price SD")
ax.legend()
plt.tight_layout()
plt.savefig(plots_output + "histogram_sd.png", dpi=150)
plt.close()


# --- Plot 4: Weighted mean price by round (line chart) ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(round_summary["round"], round_summary["weighted_mean_price"],
        marker="o", color="steelblue", label="Weighted mean")
ax.plot(round_summary["round"], round_summary["unweighted_mean_price"],
        marker="o", linestyle="--", color="grey", label="Unweighted mean")
ax.set_xlabel("Survey Round")
ax.set_ylabel("Mean Price per kg (ETB)")
ax.set_title("Weighted Mean Maize Price per kg Across Survey Rounds")
ax.legend()
plt.tight_layout()
plt.savefig(plots_output + "mean_price_by_round.png", dpi=150)
plt.close()


# --- Plot 5: Within-round CV by round ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(within_round_cv["round"], within_round_cv["within_round_cv"],
        marker="o", color="steelblue", label="Unweighted CV")
ax.plot(within_round_cv["round"], within_round_cv["weighted_within_round_cv"],
        marker="o", linestyle="--", color="grey", label="Weighted CV")
ax.set_xlabel("Survey Round")
ax.set_ylabel("Coefficient of Variation")
ax.set_title("Within-Round Maize Price Dispersion Across Households")
ax.legend()
plt.tight_layout()
plt.savefig(plots_output + "within_round_cv.png", dpi=150)
plt.close()



hh_stats.to_csv(output_file, index=False)

