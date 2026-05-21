"""
Program Impact Evaluation Framework

A comprehensive toolkit for evaluating program and intervention impacts
using various statistical and econometric methods.
"""

__version__ = "0.1.0"
__author__ = "Wainaina6943"

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.analysis import StatisticalAnalyzer
from src.evaluation import ImpactEvaluator
from src.visualization import Visualizer

__all__ = [
    "DataLoader",
    "DataPreprocessor",
    "StatisticalAnalyzer",
    "ImpactEvaluator",
    "Visualizer",
]
