"""Shared, auditable prediction path for the restricted pre-disbursement prototype."""
from pathlib import Path
import joblib
import pandas as pd

MODEL_RELATIVE = Path('experiments/temporal_robustness_8gb_2026_10_21/pre_disbursement_conditional_pipeline.joblib')

def load_bundle(project_root):
    path = Path(project_root) / MODEL_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f'Model not found: {path}')
    bundle = joblib.load(path)  # Load only your own trusted locally trained model.
    required = {'pipeline', 'threshold', 'source_fields', 'numeric_fields', 'categorical_fields'}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError('Unexpected model bundle structure')
    return bundle

def frame_from_inputs(bundle, inputs):
    missing = [f for f in bundle['source_fields'] if f not in inputs]
    if missing:
        raise ValueError(f'Missing inputs: {missing}')
    frame = pd.DataFrame([{f: inputs[f] for f in bundle['source_fields']}], columns=bundle['source_fields'])
    for field in bundle['numeric_fields']:
        frame[field] = pd.to_numeric(frame[field], errors='coerce')
    for field in bundle['categorical_fields']:
        frame[field] = frame[field].astype('string').fillna('')
    return frame

def predict_inputs(bundle, inputs):
    frame = frame_from_inputs(bundle, inputs)
    score = float(bundle['pipeline'].predict_proba(frame)[0, 1])
    return {'score': score, 'flag': bool(score >= float(bundle['threshold'])), 'threshold': float(bundle['threshold'])}
