"""
data_utils.py
-------------
Generates a synthetic dataset that mirrors the structure and statistical
distributions of the Kaggle "Data Science Job Salaries" dataset, then
provides all feature-engineering helpers used by both the trainer and the
Streamlit app.

Why synthetic data?
    The Kaggle dataset cannot be bundled with the repo due to licence
    restrictions, so we reproduce its key statistical properties from
    publicly-available summaries (medians, IQRs, job-title distributions).
    Reproducibility is ensured by a fixed random seed.
"""

import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42

# Experience levels map to an ordinal integer used as a feature.
# EN = Entry (0-2 yrs), MI = Mid (2-5), SE = Senior (5-10), EX = Executive (10+)
EXPERIENCE_LEVELS = ["EN", "MI", "SE", "EX"]
EXPERIENCE_LABEL  = {
    "EN": "Entry-Level",
    "MI": "Mid-Level",
    "SE": "Senior",
    "EX": "Executive / Director",
}
EXPERIENCE_ORDER = {"EN": 0, "MI": 1, "SE": 2, "EX": 3}

# Employment types and their labels
EMPLOYMENT_TYPES = ["FT", "CT", "PT", "FL"]
EMPLOYMENT_LABEL = {
    "FT": "Full-Time",
    "CT": "Contract",
    "PT": "Part-Time",
    "FL": "Freelance",
}

# Job categories (we collapse 150+ raw titles into 7 meaningful groups)
JOB_CATEGORIES = [
    "Data Scientist",
    "ML / AI Engineer",
    "Data Engineer",
    "Data Analyst",
    "Research Scientist",
    "BI / Analytics Engineer",
    "Management / Leadership",
]

# Company locations grouped into broad regions (reduces cardinality)
REGIONS = ["United States", "Europe", "United Kingdom", "Asia / Pacific", "Other"]
REGION_SHORT = {
    "United States":   "US",
    "Europe":          "EU",
    "United Kingdom":  "UK",
    "Asia / Pacific":  "APAC",
    "Other":           "Other",
}

# Company sizes
COMPANY_SIZES = ["S", "M", "L"]
COMPANY_SIZE_LABEL = {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (250+)"}
COMPANY_SIZE_ORDER = {"S": 0, "M": 1, "L": 2}

# Remote ratios (0 = on-site, 50 = hybrid, 100 = fully remote)
REMOTE_VALUES = [0, 50, 100]
REMOTE_LABEL = {0: "On-Site", 50: "Hybrid", 100: "Fully Remote"}

# ── Base salary tables (USD, approximate medians from public DS salary reports)──
# These form the "signal" that our synthetic noise is added on top of.
# Rows = experience, Cols = region.
BASE_SALARY = {
    #              US       EU       UK     APAC    Other
    "EN": {  "US": 85_000, "EU": 45_000, "UK": 42_000, "APAC": 35_000, "Other": 30_000 },
    "MI": {  "US":115_000, "EU": 62_000, "UK": 58_000, "APAC": 50_000, "Other": 42_000 },
    "SE": {  "US":155_000, "EU": 85_000, "UK": 80_000, "APAC": 70_000, "Other": 58_000 },
    "EX": {  "US":210_000, "EU":115_000, "UK":105_000, "APAC": 90_000, "Other": 75_000 },
}

# Multipliers applied on top of the base salary for each job category
CATEGORY_MULTIPLIER = {
    "Data Scientist":           1.00,
    "ML / AI Engineer":         1.18,   # ML engineers command a premium
    "Data Engineer":            1.05,
    "Data Analyst":             0.82,
    "Research Scientist":       1.12,
    "BI / Analytics Engineer":  0.88,
    "Management / Leadership":  1.25,
}

# Company size adjustments
SIZE_ADJUSTMENT = {"S": -8_000, "M": 5_000, "L": 18_000}

# Remote-work adjustment (fully remote roles at US companies can pay globally)
REMOTE_ADJUSTMENT = {0: 0, 50: 2_000, 100: 4_000}

# Employment-type adjustments
EMPLOYMENT_ADJUSTMENT = {"FT": 0, "CT": 10_000, "PT": -30_000, "FL": -5_000}

# ── Feature names used in the ML pipeline ─────────────────────────────────────

# Raw (human-readable) feature names displayed in the UI
FEATURE_DISPLAY_NAMES = {
    "experience_enc":    "Experience Level",
    "category_enc":      "Job Category",
    "region_enc":        "Company Region",
    "size_enc":          "Company Size",
    "remote_ratio":      "Remote Work Ratio",
    "employment_enc":    "Employment Type",
    "is_us":             "US-Based Company",
    "is_large":          "Large Company",
    "exp_x_us":          "Experience × US Company",   # interaction term
    "exp_x_category":    "Experience × Job Category", # interaction term
}

FEATURE_NAMES = list(FEATURE_DISPLAY_NAMES.keys())


# ── Data generation ────────────────────────────────────────────────────────────

def generate_dataset(n_samples: int = 4_000) -> pd.DataFrame:
    """
    Build a synthetic data science jobs dataset with realistic salary
    distributions.  The same seed is always used so the 'dataset' is
    deterministic — every run produces the same CSV.

    Parameters
    ----------
    n_samples : int
        Number of job-posting rows to generate.

    Returns
    -------
    pd.DataFrame with columns:
        experience_level, employment_type, job_category,
        company_region, company_size, remote_ratio, salary_usd
    """
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Draw categorical features from realistic marginal distributions ------

    experience_level = rng.choice(
        EXPERIENCE_LEVELS,
        size=n_samples,
        # Distribution roughly mirrors the Kaggle dataset:
        # ~25% entry, 35% mid, 30% senior, 10% executive
        p=[0.25, 0.35, 0.30, 0.10],
    )

    employment_type = rng.choice(
        EMPLOYMENT_TYPES,
        size=n_samples,
        p=[0.82, 0.10, 0.04, 0.04],  # overwhelming majority FT
    )

    job_category = rng.choice(
        JOB_CATEGORIES,
        size=n_samples,
        p=[0.25, 0.20, 0.20, 0.15, 0.08, 0.07, 0.05],
    )

    company_region = rng.choice(
        REGIONS,
        size=n_samples,
        p=[0.52, 0.23, 0.10, 0.08, 0.07],
    )

    company_size = rng.choice(
        COMPANY_SIZES,
        size=n_samples,
        p=[0.15, 0.45, 0.40],
    )

    remote_ratio = rng.choice(REMOTE_VALUES, size=n_samples, p=[0.30, 0.30, 0.40])

    # --- Compute salary with structured noise ---------------------------------

    salaries = np.zeros(n_samples)
    for i in range(n_samples):
        exp   = experience_level[i]
        reg   = REGION_SHORT[company_region[i]]
        cat   = job_category[i]
        size  = company_size[i]
        rem   = remote_ratio[i]
        emp   = employment_type[i]

        base = BASE_SALARY[exp][reg]
        salary = (
            base
            * CATEGORY_MULTIPLIER[cat]
            + SIZE_ADJUSTMENT[size]
            + REMOTE_ADJUSTMENT[rem]
            + EMPLOYMENT_ADJUSTMENT[emp]
        )

        # Add realistic log-normal noise (~15% std of base) so that the
        # distribution looks like real salary data rather than a grid.
        noise_pct = rng.normal(0, 0.15)
        salary = salary * (1 + noise_pct)

        # Clip to plausible range — salaries below ~$18K or above $500K are rare
        salaries[i] = np.clip(salary, 18_000, 500_000)

    df = pd.DataFrame({
        "experience_level": experience_level,
        "employment_type":  employment_type,
        "job_category":     job_category,
        "company_region":   company_region,
        "company_size":     company_size,
        "remote_ratio":     remote_ratio.astype(int),
        "salary_usd":       salaries.round(0).astype(int),
    })

    return df


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the raw categorical columns into numeric features ready for
    tree-based models.  Returns a copy of the dataframe with only the columns
    listed in FEATURE_NAMES plus 'salary_usd' (when present).

    Why these encodings?
    - Ordinal encoding for experience / size: the natural ordering carries
      real information that label-encoding preserves cheaply.
    - Label encoding for category / region: tree splits don't care about
      ordering for these, and one-hot would add 10+ sparse columns.
    - Interaction terms (exp × US, exp × category) capture the well-known
      fact that seniority premiums are much larger at US tech firms.
    """
    out = df.copy()

    # Ordinal encodings (preserves inherent order)
    out["experience_enc"] = out["experience_level"].map(EXPERIENCE_ORDER)
    out["size_enc"]        = out["company_size"].map(COMPANY_SIZE_ORDER)

    # Label encodings (no ordinal meaning, but trees handle this fine)
    out["category_enc"]   = out["job_category"].map(
        {c: i for i, c in enumerate(JOB_CATEGORIES)}
    )
    out["region_enc"]     = out["company_region"].map(
        {r: i for i, r in enumerate(REGIONS)}
    )
    out["employment_enc"] = out["employment_type"].map(
        {e: i for i, e in enumerate(EMPLOYMENT_TYPES)}
    )

    # Binary helpers — trees love these for clean primary splits
    out["is_us"]    = (out["company_region"] == "United States").astype(int)
    out["is_large"] = (out["company_size"] == "L").astype(int)

    # Interaction terms
    out["exp_x_us"]       = out["experience_enc"] * out["is_us"]
    out["exp_x_category"] = out["experience_enc"] * out["category_enc"]

    # Select only the model features (+ target if present)
    cols = FEATURE_NAMES + (["salary_usd"] if "salary_usd" in out.columns else [])
    return out[cols]


def build_input_row(
    experience_level: str,
    job_category: str,
    company_region: str,
    company_size: str,
    remote_ratio: int,
    employment_type: str,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame from user inputs and run it through
    engineer_features so it's ready for model.predict().
    """
    row = pd.DataFrame([{
        "experience_level": experience_level,
        "employment_type":  employment_type,
        "job_category":     job_category,
        "company_region":   company_region,
        "company_size":     company_size,
        "remote_ratio":     remote_ratio,
    }])
    return engineer_features(row)
