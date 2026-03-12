import pandas as pd
variability = pd.read_csv("/users/davisturner/desktop/PriceCSVs/maize_variability.csv", low_memory=False)
r11 = pd.read_csv("/users/davisturner/desktop/MobilePhone/wb_lsms_hfpm_hh_survey_round11_clean_microdata.csv", low_memory=False)
r11_signal = r11[["household_id", "pho3_signal", "pho4_bars"]].copy()
r11_signal["pho3_signal"] = r11_signal["pho3_signal"].replace(-98, pd.NA)
r11_signal["pho4_bars"]   = r11_signal["pho4_bars"].replace(-98, pd.NA)
merged = variability.merge(r11_signal, on="household_id", how="left")
merged.to_csv("/users/davisturner/desktop/PriceCSVs/maize_variability_signal.csv", index=False)
