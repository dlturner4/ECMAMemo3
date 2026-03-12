import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


panel = pd.read_csv("/users/davisturner/desktop/PriceCSVs/1319price.csv", low_memory=False)
# sets outliers
panel = panel[panel["outlier"] == 0].copy()


round_counts = panel.groupby("household_id")["round"].count()
multi_round = round_counts[round_counts > 1].index
panel = panel[panel["household_id"].isin(multi_round)].copy()


hh_stats = panel.groupby("household_id").agg(
    price_cv = ("price_per_kg", lambda x: x.std() / x.mean()),
    price_sd = ("price_per_kg", "std"),
    price_range = ("price_per_kg", lambda x: x.max() - x.min()),
    price_mean = ("price_per_kg", "mean"),
    price_min = ("price_per_kg", "min"),
    price_max = ("price_per_kg", "max"),
    rounds_count = ("round", "count"),
    sampling_weight = ("sampling_weight", "last")
).reset_index()


weights = hh_stats["sampling_weight"]
cv_values = hh_stats["price_cv"]
sd_values = hh_stats["price_sd"]
range_values = hh_stats["price_range"]

weighted_mean_cv = np.average(cv_values, weights=weights)
weighted_mean_range = np.average(range_values, weights=weights)
weighted_mean_sd = np.average(sd_values, weights=weights)

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


fig, ax = plt.subplots(figsize=(10, 6))
round_groups = [panel[panel["round"] == r]["price_per_kg"].values
                for r in sorted(panel["round"].unique())]
ax.boxplot(round_groups, labels=sorted(panel["round"].unique()))
ax.set_xlabel("Survey Round")
ax.set_ylabel("Price per kg (ETB)")
ax.set_title("Distribution of Maize Price per kg by Survey Round")
plt.tight_layout()
plt.savefig("/users/davisturner/desktop/PriceCSVs/" + "boxplot_by_round.png", dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(cv_values, bins=40, edgecolor="black", color="steelblue")
ax.axvline(weighted_mean_cv, color="red", linestyle="--",
           label=f"Weighted mean CV: {weighted_mean_cv:.3f}")
ax.set_xlabel("Coefficient of Variation")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Within-Household Maize Price Variability (CV)")
ax.legend()
plt.tight_layout()
plt.savefig("/users/davisturner/desktop/PriceCSVs/" + "histogram_cv.png", dpi=150)
plt.close()



fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(sd_values, bins=40, edgecolor="black", color="steelblue")
ax.axvline(weighted_mean_sd, color="red", linestyle="--",
           label=f"Weighted mean SD: {weighted_mean_sd:.2f} ETB/kg")
ax.set_xlabel("Standard Deviation (ETB/kg)")
ax.set_ylabel("Number of Households")
ax.set_title("Distribution of Within-Household Maize Price SD")
ax.legend()
plt.tight_layout()
plt.savefig("/users/davisturner/desktop/PriceCSVs/" + "histogram_sd.png", dpi=150)
plt.close()



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
plt.savefig("/users/davisturner/desktop/PriceCSVs/" + "mean_price_by_round.png", dpi=150)
plt.close()


hh_stats.to_csv("/users/davisturner/desktop/PriceCSVs/maize_variability.csv", index=False)

