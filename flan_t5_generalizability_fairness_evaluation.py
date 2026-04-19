#!/usr/bin/env python
"""
Generalizability and Fairness Evaluation.
This script performs Multisite generalizability and fairness evaluation of FLAN-T5-Large.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    f1_score,
    accuracy_score
)

# Normalize race
def normalize_race(x):
    if pd.isna(x):
        return None
    x = str(x).lower().strip()
    return x

RACE_MAP = {
    "white": "White",
    "caucasian": "White",
    "black": "Black or African American",
    "african american": "Black or African American",
    "african-american": "Black or African American",
    "black or african american": "Black or African American",
    "african american or black": "Black or African American",
    "black/african american": "Black or African American",
    "asian": "Asian",
    "chinese": "Asian",
    "japanese": "Asian",
    "korean": "Asian",
    "indian": "Asian",
    "american indian": "American Indian or Alaska Native",
    "alaska native": "American Indian or Alaska Native",
    "native american": "American Indian or Alaska Native",
    "native hawaiian": "Native Hawaiian or Other Pacific Islander",
    "Native Hawaiian/Other Pacific Islander": "Native Hawaiian or Other Pacific Islander",
    "pacific islander": "Native Hawaiian or Other Pacific Islander",
    "samoan": "Native Hawaiian or Other Pacific Islander",
    "multiple": "Two or More Races",
    "multiple races": "Two or More Races",
    "hispanic": "Hispanic or Latino",
    "other": "Other races",
    "other/unknown": "Other races",
    "unable to determine or unknown": "Other races",
    "unavailable": "Other races"
}

def standardize_race(value):
    norm = normalize_race(value)
    if not norm:
        return "Other races"
    for key, mapped in RACE_MAP.items():
        if key in norm:
            return mapped
    return "Other races"

# Normalize Note type
def normalize_note_type(x):
    if pd.isna(x):
        return None
    x = str(x).upper().strip()
    return x

NOTE_TYPE_MAP = {
    # H&P
    "HISTORY AND PHYSICAL EXAMINATION": "H&P",
    "HISTORY AND PHYSICAL": "H&P",
    "HISTORY AND PHYSICAL UPDATE": "H&P",
    "HISTORY OBTAINED FROM": "H&P",
    # Discharge
    "DISCHARGE SUMMARY": "DISCHARGE SUMMARY",
    "DEPART SUMMARY": "DISCHARGE SUMMARY",
    "BEH DEPART PATIENT SUMMARY": "DISCHARGE SUMMARY",
    "OT DISCHARGE SUMMARY": "DISCHARGE SUMMARY",
    "ED MD SUMMARY": "DISCHARGE SUMMARY",
    "CM DISCHARGE PLAN NOTE": "DISCHARGE SUMMARY",
    # Progress
    "PROGRESS NOTE": "PROGRESS NOTE",
    "MAPS PROGRESS NOTE": "PROGRESS NOTE",
    "ELECTROPHYSIOLOGY PROGRESS NOTE": "PROGRESS NOTE",
    "PSYCHIATRY PROGRESS NOTE": "PROGRESS NOTE",
    "MEDICAL RESPONSE TEAM NOTE": "PROGRESS NOTE",
    "FINAL REPORT": "PROGRESS NOTE",
    "PRELIMINARY REPORT": "PROGRESS NOTE",
    # Social work / care management
    "MEDICAL SOCIAL WORK NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    "BEH MEDICAL SOCIAL WORK NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    "CASE MANAGEMENT NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    "CARE MANAGEMENT NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    "CARE MANAGEMENT NOTE TYPE": "SOCIAL WORK/CARE MANAGEMENT",
    "HME REFERRAL NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    "PATIENT NAVIGATOR NOTE": "SOCIAL WORK/CARE MANAGEMENT",
    # Patient-reported
    "PATIENT REPORTED PROBLEMS OR SYMPTOMS": "PATIENT-REPORTED",
    "PATIENT ENTERED CLIPBOARD NOTE": "PATIENT-REPORTED",
    # ED/nursing/admin
    "ED NURSING NOTE": "ED/NURSING/ADMIN",
    "EDT NOTE": "ED/NURSING/ADMIN",
    "ED DISPOSITION NOTE": "ED/NURSING/ADMIN",
    "CODING SUMMARY": "ED/NURSING/ADMIN",
    # Therapy / allied health
    "OCCUPATIONAL THERAPY NOTE": "THERAPY/ALLIED HEALTH",
    "OT NOTE": "THERAPY/ALLIED HEALTH",
    "OT EVALUATION": "THERAPY/ALLIED HEALTH",
    "OT PROGRESS NOTE": "THERAPY/ALLIED HEALTH",
    "OT TREATMENT NOTE": "THERAPY/ALLIED HEALTH",
    "PHYSICAL THERAPY NOTE": "THERAPY/ALLIED HEALTH",
    "SPEECH THERAPY NOTE": "THERAPY/ALLIED HEALTH",
}

def standardize_note_type(value):
    norm = normalize_note_type(value)
    if not norm:
        return "Other"
    for key, mapped in NOTE_TYPE_MAP.items():
        if key in norm:
            return mapped
    return "Other"

# Categorize age
def categorize_age(df, age_column="AGE_AT_VISIT"):
    """
    Categorizes ages into age groups: 50–64, 65–79, 80+.
    """
    bins = [49, 64, 79, np.inf]
    labels = ["50–64", "65–79", "80+"]
    age_groups = pd.cut(
        df[age_column],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    return age_groups

def aggregate_small_groups_race(df, group_col, min_n=50, other_label="Other races"):
    df = df.copy()
    group_counts = df[group_col].value_counts(dropna=False)
    small_groups = group_counts[group_counts < min_n].index
    df[group_col] = df[group_col].apply(lambda x: other_label if x in small_groups else x)
    return df

def aggregate_small_groups_note_type(df, group_col, min_n=50, other_label="Other"):
    if df.empty:
        return df
    df = df.copy()
    group_counts = df[group_col].value_counts(dropna=False)
    small_groups = group_counts[group_counts < min_n].index
    df[group_col] = df[group_col].apply(lambda x: other_label if x in small_groups else x)
    return df

# Multisite generalizability and fairness evaluation.
def evaluate_generalizability(
    df: pd.DataFrame,
    output_dir: str = "/results/",
    positive_class: str = "social isolation",
):
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    REQUIRED_COLS = ["ACTUAL_LABEL", "PREDICTION", "SITE"]
    assert all(c in df.columns for c in REQUIRED_COLS)
    classes = sorted(df["ACTUAL_LABEL"].unique())
    sites = sorted(df["SITE"].unique())
    STRATA = ["SEX", "AGE_GROUP", "RACE_OMB", "SVI", "NOTE_TYPE"]

    # Helpers Functions
    def macro_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    def class_counts(y):
        vc = y.value_counts()
        return {
            "n": len(y),
            "n_si": vc.get("social isolation", 0),
            "n_ss": vc.get("social support", 0),
            "n_nsr": vc.get("no social reference", 0),
        }

    def clf_report_df(y_true, y_pred, site=None):
        rpt = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        df_rpt = pd.DataFrame(rpt).T.reset_index().rename(columns={"index": "class"})
        if site:
            df_rpt["site"] = site
        return df_rpt
    
    # Computes metrics (macro F1, accuracy, class counts) for each subgroup in `group_col`.
    def subgroup_metrics(data, group_col, min_n=50):
        df = data.copy()
        if group_col == "RACE_OMB":
            other_label = "Other races"
        else:
            other_label = "Other"

        # Count rows per group
        counts = df[group_col].value_counts()
        small_groups = counts[counts < min_n].index
    
        # Aggregate small groups
        df[group_col] = df[group_col].apply(lambda x: other_label if x in small_groups else x)
    
        # Compute metrics per group
        rows = []
        for g, gdf in df.groupby(group_col):
            rows.append({
                "group": g,
                "macro_f1": macro_f1(gdf["ACTUAL_LABEL"], gdf["PREDICTION"]),
                "accuracy": accuracy_score(gdf["ACTUAL_LABEL"], gdf["PREDICTION"]),
                **class_counts(gdf["ACTUAL_LABEL"]),
            })
        return pd.DataFrame(rows)
        
    def disparity(df_metrics):
        if len(df_metrics) < 2:
            return np.nan
        return df_metrics["macro_f1"].max() - df_metrics["macro_f1"].min()
        
    # Computes group-level positive prediction rates and SPD.
    def statistical_parity_difference(df, group_col, positive_label):
        rows = []
        for g in df[group_col].dropna().unique():
            sub = df[df[group_col] == g]
            if len(sub) == 0:
                continue
            positive_rate = (sub["PREDICTION"] == positive_label).mean()
            rows.append({
                group_col: g,
                "positive_prediction_rate": positive_rate,
                "n": len(sub),
            })
        rates_df = pd.DataFrame(rows)
        if len(rates_df) < 2:
            return rates_df, np.nan
    
        spd = (rates_df["positive_prediction_rate"].max() - rates_df["positive_prediction_rate"].min())    
        return rates_df, float(spd)

    # Computes group-level true positive rates and EOD.    
    def equal_opportunity_difference(df, group_col, positive_label):    
        rows = []
        for g in df[group_col].dropna().unique():
            sub = df[df[group_col] == g]
            y_true = (sub["ACTUAL_LABEL"] == positive_label).astype(int)
            y_pred = (sub["PREDICTION"] == positive_label).astype(int)
    
            # Skip groups with no positives in ground truth
            if y_true.sum() == 0:
                continue
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            tpr = tp / (tp + fn)
            rows.append({group_col: g, "true_positive_rate": tpr, "n": len(sub), })
        tprs_df = pd.DataFrame(rows)
        if len(tprs_df) < 2:
            return tprs_df, np.nan
        eod = (tprs_df["true_positive_rate"].max()- tprs_df["true_positive_rate"].min())
        return tprs_df, float(eod)

    # Site-level performance. - Table 1
    site_reports = []
    for site in sites:
        sdf = df[df["SITE"] == site]
        site_reports.append(clf_report_df(sdf["ACTUAL_LABEL"], sdf["PREDICTION"], site))
    pd.concat(site_reports).to_csv(f"{output_dir}/classification_report_by_site.csv", index=False)

    # Pooled subgroup performance - Table 2
    pooled_rows = []
    for s in STRATA:
        if s not in df.columns:
            continue
        sm = subgroup_metrics(df, s)
        if sm.empty:
            continue

        pooled_rows.append({
            "stratum": s,
            "f1_disparity": disparity(sm),
            "groups": len(sm),
            "total_n": sm["n"].sum(),
            "total_n_si": sm["n_si"].sum(),
            "total_n_ss": sm["n_ss"].sum(),
            "total_n_nsr": sm["n_nsr"].sum(),
        })
        sm.assign(stratum=s).to_csv(f"{output_dir}/pooled_subgroups_{s.lower()}.csv", index=False)

    pooled_fairness = pd.DataFrame(pooled_rows)
    pooled_fairness.to_csv(f"{output_dir}/pooled_fairness_summary.csv", index=False)

    # POOLED FAIRNESS (SPD & EOD) - Table 4
    fairness_rows = []    
    for s in STRATA:
        if s not in df.columns or df[s].notna().sum() < 50:
            continue    
        spd_df, spd = statistical_parity_difference(df, s, positive_class)
        eod_df, eod = equal_opportunity_difference(df, s, positive_class)
        if len(spd_df) >= 2 and len(eod_df) >= 2:
            fairness_rows.append({
                "stratum": s,
                "SPD": round(spd, 3),
                "EOD": round(eod, 3),
            })
            spd_df.to_csv(f"{output_dir}/pooled_spd_{s.lower()}.csv", index=False)
            eod_df.to_csv(f"{output_dir}/pooled_eod_{s.lower()}.csv", index=False)
    pooled_spd_eod = pd.DataFrame(fairness_rows)
    pooled_spd_eod.to_csv(f"{output_dir}/pooled_spd_eod_summary.csv", index=False)

    # WITHIN-SITE FAIRNESS ANALYSIS - Table S3
    print("\nAnalyzing within-site macro-F1 disparities")
    within_site_rows = []    
    for site in sites:
        sdf = df[df["SITE"] == site]
        if len(sdf) < 50:
            continue
        print(f"\nSite: {site} (n={len(sdf)})")
        for stratum in ["SEX", "AGE_GROUP", "RACE_OMB", "SVI", "NOTE_TYPE"]:
            if stratum not in sdf.columns:
                continue
                
            # Compute macro-F1 per subgroup
            subgroup_f1 = (
                sdf
                .groupby(stratum)
                .apply(lambda x: f1_score(
                    x["ACTUAL_LABEL"],
                    x["PREDICTION"],
                    average="macro",
                    zero_division=0
                )).dropna()
            )
            if len(subgroup_f1) < 2:
                continue
            max_f1 = subgroup_f1.max()
            min_f1 = subgroup_f1.min()

            # Subgroup labels achieving max/min F1 (handles ties)
            max_groups = subgroup_f1[subgroup_f1 == max_f1].index
            min_groups = subgroup_f1[subgroup_f1 == min_f1].index
            
            # Count records in those subgroups
            n_max_records = sdf[sdf[stratum].isin(max_groups)].shape[0]
            n_min_records = sdf[sdf[stratum].isin(min_groups)].shape[0]

            # Representative group names for display
            max_group = ", ".join(map(str, max_groups))
            min_group = ", ".join(map(str, min_groups))
    
            disparity = max_f1 - min_f1
            within_site_rows.append({
                "site": site,
                "stratum": stratum,
                "f1_disparity": round(disparity, 3),
                "max_macro_f1": round(max_f1, 3),
                "min_macro_f1": round(min_f1, 3),
                "max_subgroup": max_group,
                "min_subgroup": min_group,
                "n_max_subgroup": n_max_records,
                "n_min_subgroup": n_min_records,
                "n_groups": len(subgroup_f1),
            })
    
            print(
                f"  {stratum}: "
                f"{max_group} ({max_f1:.3f}) − "
                f"{min_group} ({min_f1:.3f}) = "
                f"{disparity:.3f}"
            )
    
    # Save explainable table
    within_site_df = pd.DataFrame(within_site_rows)
    within_site_df.to_csv(f"{output_dir}/within_site_macro_f1_disparities_explainable.csv",index=False)

    # WITHIN-SITE SPD & EOD - Table S4
    within_site_fairness = []
    for site in sites:
        sdf = df[df["SITE"] == site]
        if len(sdf) < 50:
            continue
    
        for s in STRATA:
            if s not in sdf.columns or sdf[s].notna().sum() < 50:
                continue
    
            spd_df, spd = statistical_parity_difference(sdf, s, positive_class)
            eod_df, eod = equal_opportunity_difference(sdf, s, positive_class)

            # SPD & EOD
            if len(spd_df) >= 2 and len(eod_df) >= 2:
                within_site_fairness.append({
                    "site": site,
                    "stratum": s,
                    "SPD": round(spd, 3),
                    "EOD": round(eod, 3),
                })
                spd_df.to_csv(f"{output_dir}/{site}_spd_{s.lower()}.csv",index=False)
                eod_df.to_csv(f"{output_dir}/{site}_eod_{s.lower()}.csv",index=False)
    
    within_site_spd_eod = pd.DataFrame(within_site_fairness)
    within_site_spd_eod.to_csv(f"{output_dir}/within_site_spd_eod_summary.csv",index=False)

    # WITHIN-SITE PPR (SPD attribution) - Table S5
    within_site_fairness_ppr = []
    
    for site in sites:
        sdf = df[df["SITE"] == site]
        if len(sdf) < 50:
            continue
    
        for s in STRATA:
            if s not in sdf.columns or sdf[s].notna().sum() < 50:
                continue
    
            # Compute PPR per subgroup (WITHIN SITE)
            spd_df = (
                sdf
                .groupby(s)
                .apply(lambda x: (x["PREDICTION"] == positive_class).mean())
                .reset_index(name="PPR")
                .dropna()
            )
    
            if len(spd_df) < 2:
                continue
    
            max_ppr = spd_df["PPR"].max()
            min_ppr = spd_df["PPR"].min()
            spd = max_ppr - min_ppr
    
            # Subgroups driving SPD (tie-safe)
            advantaged_groups = spd_df.loc[spd_df["PPR"] == max_ppr, s]
            disadvantaged_groups = spd_df.loc[spd_df["PPR"] == min_ppr, s]

            # Record counts in each driving subgroup
            n_advantaged = sdf[sdf[s].isin(advantaged_groups)].shape[0]
            n_disadvantaged = sdf[sdf[s].isin(disadvantaged_groups)].shape[0]
    
            within_site_fairness_ppr.append({
                "site": site,
                "stratum": s,
                "SPD": round(spd, 3),
                "advantaged_subgroup": ", ".join(map(str, advantaged_groups)),
                "disadvantaged_subgroup": ", ".join(map(str, disadvantaged_groups)),
                "n_advantaged_records": n_advantaged,
                "n_disadvantaged_records": n_disadvantaged,
            })

    within_site_fairness_ppr_df = pd.DataFrame(within_site_fairness_ppr)
    within_site_fairness_ppr_df.to_csv(f"{output_dir}/within_site_spd_site_ppr.csv", index=False)

    # Return
    return {
        "pooled_fairness": pooled_fairness,
        "pooled_spd_eod": pooled_spd_eod,
        "within_site_fairness": within_site_df,
        "within_site_spd_eod": within_site_spd_eod,
    }

def main():
    # Read data
    df_site1_src = pd.read_csv("/data/data_site1_subset.csv")
    df_site2_src = pd.read_csv("/data/data_site2_subset.csv")
    df_site3_src = pd.read_csv("/data/data_site3_subset.csv")

    # Add Site
    df_site1_src['SITE']='site1'
    df_site2_src['SITE']='site2'
    df_site3_src['SITE']='site3'

    # Normalize Race
    df_site1_src["RACE_OMB"] = df_site1_src["RACE"].apply(standardize_race)
    df_site2_src["RACE_OMB"] = df_site2_src["RACE"].apply(standardize_race)
    df_site3_src["RACE_OMB"] = df_site3_src["RACE"].apply(standardize_race)

    # Normalize Note Type
    df_site1_src["NOTE_TYPE"] = df_site1_src["CLINICAL_NOTE_TYPE"].apply(standardize_note_type)
    df_site2_src["NOTE_TYPE"] = df_site2_src["CLINICAL_NOTE_TYPE"].apply(standardize_note_type)
    df_site3_src["NOTE_TYPE"] = df_site3_src["CLINICAL_NOTE_TYPE"].apply(standardize_note_type)

    # Categorize Age
    df_site1_src["AGE_GROUP"] = categorize_age(df_site1_src)
    df_site2_src["AGE_GROUP"] = categorize_age(df_site2_src)
    df_site3_src["AGE_GROUP"] = categorize_age(df_site3_src)

    # SVI
    df_svi = pd.read_csv("/data/SVI.csv")

    # Add SVI
    df_site1 = df_site1_src.merge(df_svi[['PERSON_ID', 'SVI']], on='PERSON_ID', how="left")
    df_site2 = df_site2_src.merge(df_svi[['PERSON_ID', 'SVI']], on='PERSON_ID', how="left")
    df_site3 = df_site3_src.merge(df_svi[['PERSON_ID', 'SVI']], on='PERSON_ID', how="left")

    # Combined dataset - df_site1, df_site2, df_site3
    df_overall = pd.concat([df_site1, df_site2, df_site3], axis=0).drop_duplicates(subset="CLINICAL_NOTE_ID")
    df_overall = aggregate_small_groups_race(df_overall, 'RACE_OMB')
    df_overall = aggregate_small_groups_note_type(df_overall, 'NOTE_TYPE')

    # Evaluate
    results = evaluate_generalizability(df_overall)

if __name__ == "__main__":
    main()