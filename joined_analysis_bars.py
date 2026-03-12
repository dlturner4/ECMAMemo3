import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
input_csv   = "/users/davisturner/desktop/PriceCSVs/maize_variability_signal.csv"
output_bar   = "/users/davisturner/desktop/PriceCSVs/signal_vs_cv_bar.png"
output_scatter = "/users/davisturner/desktop/PriceCSVs/scatter_signal_cv.png"
# ============================================================

df = pd.read_csv(INPUT_FILE, low_memory=False)
df_clean = df[df["pho4_bars"].notna() & df["price_cv"].notna()].copy()
df_clean = df_clean[df_clean["pho4_bars"] > 0].copy()


signal_groups = df_clean.groupby("pho4_bars").apply(
    lambda x: pd.Series({
        "weighted_mean_cv":   np.average(x["price_cv"], weights=x["sampling_weight"]),
        "unweighted_mean_cv": x["price_cv"].mean(),
        "n_households":       len(x)
    })
).reset_index()
print(signal_groups.to_string(index=False))

weights_normalized = (df_clean["sampling_weight"] / df_clean["sampling_weight"].sum() * len(df_clean)).round().astype(int)
df_expanded = df_clean.loc[df_clean.index.repeat(weights_normalized)].reset_index(drop=True)
corr, pval = spearmanr(df_expanded["pho4_bars"], df_expanded["price_cv"])

print(f"\n=== Weighted Spearman Correlation ===")
print(f"Correlation coefficient: {corr:.4f}")
print(f"P-value:                 {pval:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    signal_groups["pho4_bars"],
    signal_groups["weighted_mean_cv"],
    color="steelblue",
    edgecolor="black",
    width=0.5
)
ax.set_xlabel("Mobile Signal Strength (Bars)")
ax.set_ylabel("Weighted Mean CV of Maize Price")
ax.set_title("Maize Price Variability by Mobile Signal Strength")
ax.set_xticks(sorted(df_clean["pho4_bars"].unique()))

for _, row in signal_groups.iterrows():
    ax.text(row["pho4_bars"], row["weighted_mean_cv"] + 0.002,
            f'n={int(row["n_households"])}', ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(output_bar, dpi=150)
plt.close()
print(f"\nSaved: {OUTPUT_BAR}")


fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(
    df_clean["pho4_bars"],
    df_clean["price_cv"],
    s=df_clean["sampling_weight"] / df_clean["sampling_weight"].max() * 100,
    alpha=0.4,
    color="steelblue",
    edgecolor="none"
)

x_vals = df_clean["pho4_bars"].values
y_vals = df_clean["price_cv"].values
w_vals = df_clean["sampling_weight"].values
coeffs = np.polyfit(x_vals, y_vals, deg=1, w=w_vals)
x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
y_line = np.polyval(coeffs, x_line)
ax.plot(x_line, y_line, color="red", linewidth=2, label=f"Weighted regression (r={corr:.3f}, p={pval:.3f})")

ax.set_xlabel("Mobile Signal Strength (Bars)")
ax.set_ylabel("Household Price CV")
ax.set_title("Maize Price Variability vs Mobile Signal Strength")
ax.legend()
plt.tight_layout()
plt.savefig(output_scatter, dpi=150)
plt.close()
print(f"Saved: {OUTPUT_SCATTER}")
