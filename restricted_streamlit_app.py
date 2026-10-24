"""Standalone 12-source-field retrospective decision-support prototype.
Run from the project root: streamlit run restricted_streamlit_app.py
"""
from pathlib import Path
import streamlit as st
from restricted_prediction_core import load_bundle, predict_inputs

st.set_page_config(page_title='Loan repayment risk | restricted model', layout='centered')
st.title('Loan repayment risk — restricted pre-disbursement prototype')
st.warning('Research prototype only. Not validated for live underwriting. Scores are uncalibrated and cohort-selected; they are not established real-world default probabilities.')
st.caption('This model uses 12 source fields and its saved preprocessing pipeline. It does not reconstruct 143 hidden predictors.')

@st.cache_resource
def get_bundle():
    return load_bundle(Path(__file__).resolve().parent)

try:
    bundle = get_bundle()
except Exception as exc:
    st.error(f'Cannot load the saved model: {exc}')
    st.stop()

# Obtain actual fitted categorical vocabularies, rather than inventing input choices.
def categories_by_field(bundle):
    preprocessing = bundle['pipeline'].named_steps['preprocessing']
    result = {}
    for name, transformer, columns in preprocessing.transformers_:
        if name == 'remainder':
            continue
        encoder = transformer if hasattr(transformer, 'categories_') else None
        if encoder is None and hasattr(transformer, 'named_steps'):
            encoder = next((step for step in transformer.named_steps.values() if hasattr(step, 'categories_')), None)
        if encoder is not None:
            for field, categories in zip(columns, encoder.categories_):
                result[field] = [str(v) for v in categories if str(v).lower() not in ('nan', '<na>')]
    return result

vocab = categories_by_field(bundle)
with st.form('risk_form'):
    inputs = {}
    for field in bundle['source_fields']:
        if field in bundle['numeric_fields']:
            inputs[field] = st.number_input(field.replace('_', ' ').title(), min_value=0.0, value=0.0, step=1.0 if field != 'int_rate' else 0.1, key=field)
        else:
            choices = vocab.get(field, [])
            if choices:
                inputs[field] = st.selectbox(field.replace('_', ' ').title(), choices, key=field)
            else:
                inputs[field] = st.text_input(field.replace('_', ' ').title(), key=field)
    submitted = st.form_submit_button('Calculate model score')

if submitted:
    result = predict_inputs(bundle, inputs)
    st.metric('Uncalibrated model score', f"{result['score']:.1%}")
    st.write('Retrospective threshold flag:', '**Elevated score**' if result['flag'] else '**Below selected threshold**')
    st.caption(f"Training-selected threshold: {result['threshold']:.2f}. This is not a lending recommendation or an approved decision rule.")
    st.json({'input_fields': len(bundle['source_fields']), 'model_score': round(result['score'], 8), 'threshold_flag': result['flag']})

st.divider()
st.caption('Scope: eventually resolved accepted Lending Club loans; training issue years 2007–2013, validation 2014. Model inputs and historical labels lack independently verified decision-time provenance. No causal or fairness claims.')
