"""
feature_store package
======================
- feature_engineer : FeatureEngineerAgent — computes 50+ technical features
                     every 60 seconds and upserts to feature_store hypertable.
"""

from feature_store.feature_engineer import FeatureEngineerAgent, run_feature_engineer

__all__ = ["FeatureEngineerAgent", "run_feature_engineer"]
