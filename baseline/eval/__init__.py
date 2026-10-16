"""eval 包：检索评估与检测评估。"""
from .detection import evaluate_detection
from .retrieval import build_database, evaluate_retrieval

__all__ = ["evaluate_retrieval", "build_database", "evaluate_detection"]
