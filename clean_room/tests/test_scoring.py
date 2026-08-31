import pandas as pd
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model_v2 import apply_frozen_config, FROZEN_CONFIG

def test_frozen_config_deterministic():
    """Same input must always produce identical output scores."""
    dummy = pd.DataFrame({
        'risk_score': [1.0, 0.0, 0.5],
        'node_count': [3, 0, 5],
        'has_risky_sink': [1, 0, 0],
        'log_amount_zscore': [4.0, 1.0, 2.0],
        'betweenness': [0.001, 0.0, 0.0002],
        'max_velocity_ratio': [0.95, 0.1, 0.88],
        'retention_ratio': [0.98, 0.20, 0.50],
        'counterparty_repeat_rate': [0.05, 0.90, 0.40],
        'roundness_score': [0.80, 0.10, 0.30],
        'inflow_diversity': [20, 2, 5],
        'nighttime_ratio': [0.60, 0.05, 0.15]
    })
    
    scores_run1 = apply_frozen_config(dummy, FROZEN_CONFIG)
    scores_run2 = apply_frozen_config(dummy, FROZEN_CONFIG)
    
    np.testing.assert_array_almost_equal(scores_run1, scores_run2, decimal=6)

def test_high_risk_account_exceeds_threshold():
    """An account with multi-tiered suspicious signals must exceed the decision threshold."""
    dummy = pd.DataFrame({
        'risk_score': [1.0],
        'node_count': [3],
        'has_risky_sink': [1],
        'log_amount_zscore': [4.0],
        'betweenness': [0.001],
        'max_velocity_ratio': [0.95],
        'retention_ratio': [0.99],
        'counterparty_repeat_rate': [0.02],
        'roundness_score': [0.90],
        'inflow_diversity': [25],
        'nighttime_ratio': [0.70]
    })
    
    scores = apply_frozen_config(dummy, FROZEN_CONFIG)
    assert scores[0] >= FROZEN_CONFIG['dec_thresh'], f"Expected >= {FROZEN_CONFIG['dec_thresh']}, got {scores[0]}"

def test_legitimate_merchant_stays_safe():
    """A clean account with high repeat rate and low retention ratio must remain below manual review threshold."""
    dummy = pd.DataFrame({
        'risk_score': [0.0],
        'node_count': [0],
        'has_risky_sink': [0],
        'log_amount_zscore': [0.5],
        'betweenness': [0.0],
        'max_velocity_ratio': [0.10],
        'retention_ratio': [0.20],
        'counterparty_repeat_rate': [0.95],
        'roundness_score': [0.05],
        'inflow_diversity': [3],
        'nighttime_ratio': [0.0]
    })
    
    scores = apply_frozen_config(dummy, FROZEN_CONFIG)
    assert scores[0] < FROZEN_CONFIG['manual_thresh'], f"Expected < {FROZEN_CONFIG['manual_thresh']}, got {scores[0]}"
