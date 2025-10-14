from .base_recommender import BaseRecommender
from .knn_recommender import KNNRecommender
from .rf_recommender import RFRecommender
from .gb_recommender import GBRecommender
from .lr_recommender import LRRecommender
from .random_recommender import RandomRecommender
from .cf_recommender import CFRecommender
from .hybrid_recommender import HybridRecommender

try:
    from .surprise_recommender import SurpriseRecommender
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    SurpriseRecommender = None

__all__ = [
    'BaseRecommender',
    'KNNRecommender',
    'RFRecommender',
    'GBRecommender',
    'LRRecommender',
    'RandomRecommender',
    'CFRecommender',
    'HybridRecommender',
    'SurpriseRecommender',
    'SURPRISE_AVAILABLE',
]
