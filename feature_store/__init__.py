"""
feature_store package
=====================
Technical indicator and ML feature computation for ATLAS.

Modules
-------
feature_engineer  – Computes 50+ features per symbol every 60 s
                    and upserts them into the feature_store hypertable.
"""

from feature_store.feature_engineer import FeatureEngineerAgent, run_feature_engineer

__all__ = ["FeatureEngineerAgent", "run_feature_engineer"]
