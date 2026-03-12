import pandas as pd
import os
import glob

rounds = [13, 14, 15, 16, 17, 18, 19]
all_dfs = []

for r in rounds:
    fname = f"wb_lsms_hfpm_hh_survey_round{r}_price_public.csv"
    fpath = os.path.join("/users/davisturner/desktop/PriceCSVs", fname)
    df = pd.read_csv(fpath, low_memory=False)
    maize = df[df["fp_01"] == "Maize grain/flour"].copy()
    maize = maize[maize["fp2_unit"].isin(["Kilogram", "Quntal"])].copy()
    
    # Converting fields to integers for easier division #
    maize["fp2_quant"] = pd.to_numeric(maize["fp2_quant"], errors="coerce")
    maize["fp3_price"] = pd.to_numeric(maize["fp3_price"], errors="coerce")
    
    # Ensuring that empty columns or negative values are not included for division errors #
    maize = maize[(maize["fp2_quant"].notna()) & (maize["fp2_quant"] > 0)].copy()
    maize = maize[maize["fp3_price"].notna()].copy()
    
    # Calcuating price/kg rather than reporting the raw price value #
    maize["price_per_kg"] = maize.apply(
        lambda row: (row["fp3_price"] / row["fp2_quant"]) / 100
        if row["fp2_unit"] == "Quntal"
        else row["fp3_price"] / row["fp2_quant"], axis = 1
    )
    
    # Applying Outlier Rule #
    quartile_1 = maize["price_per_kg"].quantile(0.25)
    quartile_3 = maize["price_per_kg"].quantile(0.75)
    IQR = quartile_3 - quartile_1
    lower = quartile_1 - 1.5 * IQR
    upper = quartile_3 + 1.5 * IQR
    maize["outlier"] = ((maize["price_per_kg"] < lower) | (maize["price_per_kg"] > upper)).astype(int)
    
    weight_col = f"phw{r}"
    if weight_col in maize.columns:
        maize = maize.rename(columns={weight_col: "sampling_weight"})

    maize["round"] = r

    print(f"Round {r}: {len(maize)} maize observations")
    all_dfs.append(maize)


panel = pd.concat(all_dfs, ignore_index=True)
panel.to_csv("/users/davisturner/desktop/PriceCSVs/1319price.csv", index=False)
