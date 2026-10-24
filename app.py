
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Explainable Loan Risk Decision Support System",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


# ================================================================
# DEPLOYMENT COMPONENT LOADING
# ================================================================

@st.cache_resource
def load_model_bundle():
    return joblib.load(BASE_DIR / "loan_risk_model_bundle.joblib")


@st.cache_data
def load_schema():
    schema_path = BASE_DIR / "artifact_input_schema.csv"
    schema = pd.read_csv(schema_path)

    if "position" in schema.columns:
        schema = schema.sort_values("position").reset_index(drop=True)

    return schema


@st.cache_data
def load_reference_profile():
    profile = pd.read_csv(BASE_DIR / "reference_profile.csv")

    # Support either:
    #   feature, reference_value
    # or a single-row wide representation.
    if {"feature", "reference_value"}.issubset(profile.columns):
        return pd.Series(
            profile["reference_value"].values,
            index=profile["feature"].astype(str),
            dtype=float,
        )

    if profile.shape[0] == 1:
        return profile.iloc[0].astype(float)

    if profile.shape[1] == 2:
        return pd.Series(
            profile.iloc[:, 1].values,
            index=profile.iloc[:, 0].astype(str),
            dtype=float,
        )

    raise ValueError("Unsupported reference_profile.csv structure.")


@st.cache_data
def load_json_file(filename):
    with open(BASE_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


bundle = load_model_bundle()
schema = load_schema()
reference_profile = load_reference_profile()
feature_architecture = load_json_file("feature_architecture.json")
deployment_config = load_json_file("deployment_config.json")
deployment_manifest = load_json_file("deployment_manifest.json")


# ================================================================
# RECOVER MODEL AND SHAP EXPLAINER
# ================================================================

def recover_component(obj, candidate_names):
    if isinstance(obj, dict):
        for name in candidate_names:
            if name in obj:
                return obj[name]

    for name in candidate_names:
        if hasattr(obj, name):
            return getattr(obj, name)

    return None


model = recover_component(
    bundle,
    [
        "final_model",
        "model",
        "classifier",
        "xgb_model",
        "xgb_classifier",
        "predictive_model",
    ],
)

explainer = recover_component(
    bundle,
    [
        "explainer",
        "shap_explainer",
        "tree_explainer",
    ],
)

if model is None:
    # Last-resort support if the joblib itself is the model.
    if hasattr(bundle, "predict_proba"):
        model = bundle

if model is None:
    raise RuntimeError(
        "The predictive model could not be recovered from "
        "loan_risk_model_bundle.joblib."
    )


# ================================================================
# CONFIGURATION
# ================================================================

def recover_threshold(config, default=0.23):
    candidates = [
        "decision_threshold",
        "threshold",
        "final_threshold",
        "FINAL_THRESHOLD",
    ]

    if isinstance(config, dict):
        for key in candidates:
            if key in config:
                try:
                    return float(config[key])
                except Exception:
                    pass

    return float(default)


DECISION_THRESHOLD = recover_threshold(deployment_config, 0.23)


def risk_band(probability):
    probability = float(probability)

    if probability < 0.10:
        return "Low"
    elif probability < 0.23:
        return "Moderate"
    elif probability < 0.40:
        return "Elevated"
    return "High"


def threshold_status(probability):
    return (
        "At or Above Threshold"
        if float(probability) >= DECISION_THRESHOLD
        else "Below Threshold"
    )


# ================================================================
# FEATURE ARCHITECTURE
# ================================================================

artifact_features = schema["feature"].astype(str).tolist()

def recover_model_features():
    if isinstance(feature_architecture, dict):

        candidate_keys = [
            "model_features",
            "trained_model_features",
            "full_model_features",
            "feature_names",
            "model_feature_order",
        ]

        for key in candidate_keys:
            value = feature_architecture.get(key)

            if isinstance(value, list) and len(value) > 0:
                return [str(v) for v in value]

        # Search one nested level if needed
        for value in feature_architecture.values():
            if isinstance(value, dict):
                for key in candidate_keys:
                    nested = value.get(key)
                    if isinstance(nested, list) and len(nested) > 0:
                        return [str(v) for v in nested]

    # Reference profile is already stored in complete model order
    return [str(v) for v in reference_profile.index.tolist()]


model_features = recover_model_features()

if len(artifact_features) != 40:
    raise RuntimeError(
        f"Expected 40 user-facing features; recovered {len(artifact_features)}."
    )

if len(model_features) != 183:
    raise RuntimeError(
        f"Expected 183 trained-model features; recovered {len(model_features)}."
    )


# ================================================================
# HUMAN-READABLE FEATURE LABELS
# ================================================================

DEFAULT_LABELS = {
    "int_rate": "Interest Rate",
    "term": "Loan Term",
    "open_rv_24m": "Revolving Accounts Opened in Last 24 Months",
    "mort_acc": "Mortgage Accounts",
    "acc_open_past_24mths": "Accounts Opened in Last 24 Months",
    "tot_hi_cred_lim": "Total High Credit Limit",
    "num_actv_rev_tl": "Active Revolving Accounts",
    "dti": "Debt-to-Income Ratio",
    "funded_amnt": "Funded Amount",
    "all_util": "Overall Credit Utilisation",
    "avg_cur_bal": "Average Current Balance",
    "installment": "Monthly Instalment",
    "total_bal_il": "Total Instalment Balance",
    "num_rev_tl_bal_gt_0": "Revolving Accounts with Balance",
    "annual_inc": "Annual Income",
    "max_bal_bc": "Maximum Bankcard Balance",
    "delinq_2yrs": "Delinquencies in Last 2 Years",
    "collections_12_mths_ex_med": "Collections in Last 12 Months",
    "loan_amnt": "Loan Amount",
    "bc_open_to_buy": "Bankcard Open-to-Buy",
    "open_il_12m": "Instalment Accounts Opened in Last 12 Months",
    "mths_since_recent_inq": "Months Since Most Recent Inquiry",
    "funded_amnt_inv": "Investor-Funded Amount",
    "num_tl_op_past_12m": "Accounts Opened in Last 12 Months",
    "total_bc_limit": "Total Bankcard Credit Limit",
    "mo_sin_old_rev_tl_op": "Age of Oldest Revolving Account",
    "inq_last_6mths": "Credit Inquiries in Last 6 Months",
    "pub_rec": "Public Records",
    "total_rev_hi_lim": "Total Revolving High Credit Limit",
    "mths_since_recent_bc": "Months Since Most Recent Bankcard Account",
    "emp_length": "Employment Length",
    "open_il_24m": "Instalment Accounts Opened in Last 24 Months",
    "percent_bc_gt_75": "Bankcards Above 75% Utilisation",
    "pub_rec_bankruptcies": "Public Record Bankruptcies",
    "open_acc_6m": "Accounts Opened in Last 6 Months",
    "bc_util": "Bankcard Utilisation",
    "num_actv_bc_tl": "Active Bankcard Accounts",
    "mths_since_last_delinq": "Months Since Last Delinquency",
    "mths_since_last_record": "Months Since Last Public Record",
    "total_acc": "Total Credit Accounts",
    "revol_bal": "Revolving Balance",
    "revol_util": "Revolving Utilisation (%)",
    "credit_history_years": "Credit History Length (Years)",
}


def get_label(feature, row=None):
    if row is not None:
        for column in ["label", "user_label", "display_label"]:
            if column in row.index:
                value = row[column]
                if pd.notna(value) and str(value).strip():
                    return str(value)

    if feature in DEFAULT_LABELS:
        return DEFAULT_LABELS[feature]

    return (
        str(feature)
        .replace("_", " ")
        .strip()
        .title()
    )


# ================================================================
# INPUT VALIDATION AND RECONSTRUCTION
# ================================================================

def validate_user_input(user_values):
    missing = [
        feature
        for feature in artifact_features
        if feature not in user_values
    ]

    if missing:
        raise ValueError(
            "Missing required input variable(s): "
            + ", ".join(missing)
        )

    unexpected = [
        feature
        for feature in user_values
        if feature not in artifact_features
    ]

    if unexpected:
        raise ValueError(
            "Unexpected input variable(s): "
            + ", ".join(unexpected)
        )

    validated = {}

    for _, row in schema.iterrows():
        feature = str(row["feature"])
        value = user_values[feature]

        try:
            value = float(value)
        except Exception:
            raise ValueError(f"{feature} must be numeric.")

        if not np.isfinite(value):
            raise ValueError(f"{feature} must be finite.")

        input_type = str(row.get("input_type", "numeric")).lower()

        # Field-aware domain validation for the retrospective DSS.
        # Do not use observed test maxima as arbitrary lending limits.
        if value < 0:
            raise ValueError(
                f"{get_label(feature, row)} must be non-negative."
            )

        count_features = {
            "mort_acc", "open_rv_24m", "acc_open_past_24mths",
            "open_acc_6m", "num_tl_op_past_12m", "open_il_12m",
            "open_il_24m", "num_actv_rev_tl", "num_rev_tl_bal_gt_0",
            "num_actv_bc_tl", "inq_last_6mths", "delinq_2yrs",
            "collections_12_mths_ex_med", "pub_rec",
            "pub_rec_bankruptcies",
        }
        if feature in count_features and not value.is_integer():
            raise ValueError(
                f"{get_label(feature, row)} must be a whole number."
            )

        if feature == "term" and value not in (36.0, 60.0):
            raise ValueError("Loan Term must be 36 or 60 months.")

        if feature == "percent_bc_gt_75" and value > 100:
            raise ValueError(
                "Percentage of Bankcards Above 75% Utilisation "
                "must not exceed 100."
            )

        if input_type == "integer":
            if not float(value).is_integer():
                raise ValueError(
                    f"{get_label(feature, row)} must be an integer."
                )
            value = int(round(value))

        validated[feature] = value

    return pd.Series(
        validated,
        index=artifact_features,
        dtype=float,
    )


def reconstruct_model_input(validated_input):
    reconstructed = reference_profile.copy()

    # Ensure full trained-model feature architecture
    reconstructed = reconstructed.reindex(model_features)

    for feature in artifact_features:
        reconstructed.loc[feature] = float(validated_input.loc[feature])

    if reconstructed.isna().any():
        missing = reconstructed.index[
            reconstructed.isna()
        ].tolist()

        raise ValueError(
            "Reconstruction produced missing trained-model values: "
            + ", ".join(map(str, missing))
        )

    reconstructed_df = pd.DataFrame(
        [reconstructed.values],
        columns=model_features,
    )

    return reconstructed_df.astype(float)


# ================================================================
# SHAP PROCESSING
# ================================================================

def generate_shap_values(model_input):
    if explainer is None:
        return None

    try:
        raw = explainer.shap_values(model_input)
    except Exception:
        try:
            raw = explainer(model_input)
        except Exception:
            return None

    if hasattr(raw, "values"):
        raw = raw.values

    if isinstance(raw, list):
        if len(raw) == 2:
            raw = raw[1]
        else:
            raw = raw[0]

    raw = np.asarray(raw)

    if raw.ndim == 3:
        # Handles binary-class explanation structures
        if raw.shape[-1] == 2:
            raw = raw[:, :, 1]
        else:
            raw = raw[:, :, 0]

    if raw.ndim == 2:
        raw = raw[0]

    raw = np.asarray(raw).reshape(-1)

    if len(raw) != len(model_features):
        return None

    return raw.astype(float)


# ================================================================
# COMPLETE PREDICTION SERVICE
# ================================================================

def predict_case(user_values):
    validated_input = validate_user_input(user_values)

    model_input = reconstruct_model_input(validated_input)

    probability = float(
        model.predict_proba(model_input)[0, 1]
    )

    binary_decision = int(
        probability >= DECISION_THRESHOLD
    )

    band = risk_band(probability)

    shap_values = generate_shap_values(model_input)

    shap_table = None
    top_factors = None

    if shap_values is not None:
        shap_table = pd.DataFrame({
            "feature": model_features,
            "feature_value": model_input.iloc[0].values,
            "shap_value": shap_values,
        })

        shap_table["absolute_shap"] = (
            shap_table["shap_value"].abs()
        )

        shap_table["direction"] = np.select(
            [
                shap_table["shap_value"] > 0,
                shap_table["shap_value"] < 0,
            ],
            [
                "Increases predicted risk",
                "Decreases predicted risk",
            ],
            default="Neutral contribution",
        )

        shap_table["Factor"] = shap_table["feature"].map(
            lambda x: get_label(x)
        )

        top_factors = (
            shap_table
            .sort_values(
                "absolute_shap",
                ascending=False
            )
            .head(10)
            .copy()
        )

        top_factors.insert(
            0,
            "Rank",
            range(1, len(top_factors) + 1),
        )

    return {
        "validated_input": validated_input,
        "model_input": model_input,
        "probability": probability,
        "probability_percent": probability * 100,
        "decision_threshold": DECISION_THRESHOLD,
        "binary_decision": binary_decision,
        "threshold_status": threshold_status(probability),
        "risk_band": band,
        "shap_values": shap_values,
        "shap_table": shap_table,
        "top_factors": top_factors,
    }


# ================================================================
# STREAMLIT USER INTERFACE
# ================================================================

st.title("Explainable Loan Risk Decision Support System")

st.caption(
    "An explainable machine-learning artefact supporting "
    "transparent loan default-risk assessment."
)

st.info(
    "This system provides model-based decision-support information. "
    "It does not make or recommend an autonomous lending decision."
)

# ----------------------------------------------------------------
# INPUT PRESENTATION ARCHITECTURE
# ----------------------------------------------------------------

INPUT_GROUPS = {
    "Loan Characteristics": [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "term",
        "int_rate",
        "installment",
    ],

    "Borrower Financial Profile": [
        "annual_inc",
        "dti",
        "emp_length",
        "avg_cur_bal",
    ],

    "Credit Account Profile": [
        "revol_bal",
        "mort_acc",
        "open_rv_24m",
        "acc_open_past_24mths",
        "open_acc_6m",
        "num_tl_op_past_12m",
        "open_il_12m",
        "open_il_24m",
        "num_actv_rev_tl",
        "num_rev_tl_bal_gt_0",
        "num_actv_bc_tl",
    ],

    "Credit Limits, Balances and Utilisation": [
        "tot_hi_cred_lim",
        "total_rev_hi_lim",
        "total_bc_limit",
        "total_bal_il",
        "max_bal_bc",
        "bc_open_to_buy",
        "all_util",
        "bc_util",
        "percent_bc_gt_75",
    ],

    "Credit History and Recent Activity": [
        "mo_sin_old_rev_tl_op",
        "mths_since_recent_bc",
        "mths_since_recent_inq",
        "inq_last_6mths",
        "revol_util",
        "credit_history_years",
    ],

    "Adverse Credit Events": [
        "delinq_2yrs",
        "collections_12_mths_ex_med",
        "pub_rec",
        "pub_rec_bankruptcies",
    ],
}


def render_input_field(feature, row):

    label = get_label(feature, row)

    input_type = str(
        row.get("input_type", "numeric")
    ).lower()

    reference_value = float(
        row.get(
            "reference_value",
            reference_profile.get(feature, 0.0),
        )
    )

    observed_min = row.get(
        "observed_test_min",
        np.nan,
    )

    observed_max = row.get(
        "observed_test_max",
        np.nan,
    )

    help_text = (
        "Observed development/test range: "
        f"{observed_min} to {observed_max}. "
        "The range is provided as empirical context and is "
        "not an automated lending boundary."
    )

    if input_type == "integer":

        return st.number_input(
            label,
            value=int(round(reference_value)),
            step=1,
            key=f"input_{feature}",
            help=help_text,
        )

    magnitude = abs(reference_value)

    if magnitude >= 100000:
        step = 1000.0
    elif magnitude >= 10000:
        step = 100.0
    elif magnitude >= 1000:
        step = 10.0
    elif magnitude >= 100:
        step = 1.0
    else:
        step = 0.1

    return st.number_input(
        label,
        value=float(reference_value),
        step=float(step),
        format="%.2f",
        key=f"input_{feature}",
        help=help_text,
    )


st.divider()

st.header("1. Case Input")

st.write(
    "Enter the borrower, loan and credit-profile variables below. "
    "The variables are organised into conceptual groups to improve "
    "usability while retaining the validated 40-variable input architecture."
)

st.caption(
    "Reference values are pre-populated for artefact demonstration. "
    "The information icons report empirical development/test ranges."
)

schema_lookup = (
    schema
    .set_index("feature", drop=False)
)

user_values = {}

with st.form("loan_risk_input_form"):

    for group_index, (group_name, group_features) in enumerate(
        INPUT_GROUPS.items()
    ):

        with st.expander(
            f"{group_name} ({len(group_features)} variables)",
            expanded=(group_index == 0),
        ):

            columns = st.columns(2)

            for i, feature in enumerate(group_features):

                if feature not in schema_lookup.index:
                    raise RuntimeError(
                        f"Input architecture references an "
                        f"unknown feature: {feature}"
                    )

                row = schema_lookup.loc[feature]

                with columns[i % 2]:
                    user_values[feature] = render_input_field(
                        feature,
                        row,
                    )

    st.caption(
        "All 40 variables are required. Submitted values are validated "
        "before reconstruction into the complete 183-variable "
        "trained-model representation."
    )

    submitted = st.form_submit_button(
        "Generate Risk Assessment",
        type="primary",
        use_container_width=True,
    )


# ================================================================
# RESULTS
# ================================================================

# Do not present a previous assessment alongside invalid current input.
# This check is for display safety; validate_user_input remains the
# authoritative submission-time validation layer.
income_widget_value = st.session_state.get("input_annual_inc")
income_invalid = (
    income_widget_value is not None
    and float(income_widget_value) < 0
)

if income_invalid:
    st.warning(
        "Annual Income cannot be negative. Correct this field before "
        "generating a risk assessment. No assessment is shown for "
        "the invalid input."
    )

if submitted and not income_invalid:

    try:
        result = predict_case(user_values)

    except Exception as exc:
        st.error(
            "Prediction could not be generated because the "
            f"submitted input did not pass validation: {exc}"
        )

    else:

        st.divider()
        st.header("2. Prediction Summary")

        st.caption(
            "The summary separates the model-estimated probability, "
            "validated threshold comparison and interpretive risk band."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Predicted Default Probability",
            f"{result['probability_percent']:.2f}%"
        )

        c2.metric(
            "Decision Threshold",
            f"{DECISION_THRESHOLD * 100:.2f}%"
        )

        c3.metric(
            "Threshold Decision",
            result["threshold_status"]
        )

        c4.metric(
            "Interpretive Risk Band",
            result["risk_band"]
        )

        st.divider()
        st.header("3. Risk Interpretation")

        probability_percent = (
            result["probability_percent"]
        )

        if result["binary_decision"] == 1:
            threshold_sentence = (
                "The predicted probability is at or above "
                f"the established {DECISION_THRESHOLD * 100:.0f}% "
                "binary decision threshold."
            )
        else:
            threshold_sentence = (
                "The predicted probability remains below "
                f"the established {DECISION_THRESHOLD * 100:.0f}% "
                "binary decision threshold."
            )

        st.write(
            f"The model estimates a default probability of "
            f"**{probability_percent:.2f}%**. "
            f"{threshold_sentence} "
            f"The observation falls within the "
            f"**{result['risk_band']}** interpretive risk band."
        )

        st.divider()
        st.header("4. Local SHAP Explanation")

        if result["top_factors"] is None:

            st.warning(
                "Prediction was generated successfully, but "
                "the exported SHAP explainer could not be evaluated "
                "for this observation."
            )

        else:

            top_display = result["top_factors"][
                [
                    "Rank",
                    "Factor",
                    "feature_value",
                    "shap_value",
                    "direction",
                ]
            ].copy()

            top_display.columns = [
                "Rank",
                "Factor",
                "Observed Value",
                "SHAP Contribution (log-odds)",
                "Effect on Predicted Risk",
            ]

            st.subheader(
                "Feature Contributions to Predicted Risk"
            )

            st.caption(
                "Positive SHAP values push the model prediction toward "
                "higher estimated default risk; negative values push the "
                "prediction toward lower estimated default risk."
            )

            shap_chart = (
                top_display[
                    [
                        "Factor",
                        "SHAP Contribution (log-odds)",
                    ]
                ]
                .set_index("Factor")
                .sort_values(
                    "SHAP Contribution (log-odds)",
                    ascending=True,
                )
            )

            st.bar_chart(
                shap_chart,
                horizontal=True,
            )

            st.caption(
                "The chart shows local SHAP contributions in raw log-odds for this "
                "observation, not percentage-point changes in default "
                "probability. Magnitude indicates contribution strength; "
                "the contributions do not establish causality."
            )

            st.subheader(
                "Detailed Local Explanation"
            )

            st.dataframe(
                top_display,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader(
                "Human-Readable Explanation Statements"
            )

            for _, row in top_display.iterrows():

                factor = row["Factor"]
                direction = row[
                    "Effect on Predicted Risk"
                ]

                if direction == "Increases predicted risk":
                    sentence = (
                        f"**{factor}** contributed toward a "
                        "higher predicted default risk for this case."
                    )

                elif direction == "Decreases predicted risk":
                    sentence = (
                        f"**{factor}** contributed toward a "
                        "lower predicted default risk for this case."
                    )

                else:
                    sentence = (
                        f"**{factor}** had a neutral local "
                        "contribution to this prediction."
                    )

                st.write("• " + sentence)

        st.divider()
        st.header("5. Decision-Support Narrative")

        st.write(
            f"The model estimates a default probability of "
            f"**{result['probability_percent']:.2f}%**. "
            f"This corresponds to the "
            f"**{result['risk_band']}** interpretive risk band. "
            f"{threshold_sentence} "
            "The accompanying SHAP explanation identifies the "
            "model features contributing most strongly toward "
            "higher or lower predicted risk for this observation."
        )

        st.divider()
        st.header("6. Governance and Interpretation Notices")

        st.warning(
            "**Decision-support use:** The prediction is intended "
            "to support human assessment and should not be treated "
            "as an autonomous lending decision."
        )

        st.info(
            "**Threshold interpretation:** The binary decision is "
            "generated using the validated fixed decision threshold. "
            "The threshold does not independently determine a real-world "
            "credit decision."
        )

        st.info(
            "**Risk-band interpretation:** Low, Moderate, Elevated "
            "and High are descriptive probability categories used "
            "to improve interpretability."
        )

        st.info(
            "**SHAP interpretation:** SHAP contributions are expressed in raw "
            "log-odds, not percentage-point changes in default probability. "
            "They describe model attribution for this prediction and "
            "do not establish causal relationships."
        )

        st.info(
            "**Model integrity:** The exported XGBoost classifier "
            "and SHAP explainer are used without retraining or "
            "modification."
        )

        st.info(
            "**Input reconstruction:** The 40 user-facing variables "
            "are reconstructed into the complete 183-variable "
            "trained-model representation before inference."
        )

        st.divider()

        with st.expander(
            "7. Technical and Traceability Details",
            expanded=False
        ):

            st.caption(
                "This section provides implementation traceability and "
                "is separated from the primary decision-support output."
            )

            st.write(
                f"User-facing feature count: "
                f"**{len(artifact_features)}**"
            )

            st.write(
                f"Trained-model feature count: "
                f"**{len(model_features)}**"
            )

            st.write(
                f"SHAP feature count: "
                f"**{len(result['shap_values']) if result['shap_values'] is not None else 'Unavailable'}**"
            )

            st.write(
                f"Decision threshold: "
                f"**{DECISION_THRESHOLD:.2f}**"
            )

            st.write(
                "Predictive model: **existing exported classifier**"
            )

            st.write(
                "Explanation pathway: **existing exported SHAP explainer**"
            )

            st.subheader(
                "Validated 40-Variable Input"
            )

            validated_display = (
                result["validated_input"]
                .rename("Value")
                .to_frame()
            )

            st.dataframe(
                validated_display,
                use_container_width=True
            )

            st.subheader(
                "Complete 183-Variable Reconstructed Input"
            )

            st.dataframe(
                result["model_input"],
                use_container_width=True
            )


st.divider()

st.caption(
    "Explainable Loan Risk Decision Support System | "
    "Prediction outputs support human decision-making. "
    "SHAP values represent model attribution rather than causation."
)
