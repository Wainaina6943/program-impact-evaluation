"""
Statistical Analysis Module

Provides methods for descriptive statistics and exploratory analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats


class StatisticalAnalyzer:
    """
    Perform statistical analysis on datasets.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the StatisticalAnalyzer.
        
        Args:
            data (pd.DataFrame, optional): Dataset to analyze
        """
        self.data = data
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the data to analyze.
        
        Args:
            data (pd.DataFrame): Dataset to analyze
        """
        self.data = data
    
    def summary_statistics(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute summary statistics.
        
        Args:
            columns (List[str], optional): Columns to analyze. If None, analyze all numeric columns.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if self.data is None:
            return pd.DataFrame()
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        return self.data[columns].describe()
    
    def compare_groups(self, outcome: str, group: str) -> Dict:
        """
        Compare outcomes across groups.
        
        Args:
            outcome (str): Outcome variable column name
            group (str): Grouping variable column name
        
        Returns:
            dict: Group statistics and test results
        """
        if self.data is None:
            return {}
        
        groups = self.data.groupby(group)[outcome].agg([
            'count', 'mean', 'std', 'min', 'max'
        ])
        
        # Perform t-test
        unique_groups = self.data[group].unique()
        if len(unique_groups) == 2:
            group1 = self.data[self.data[group] == unique_groups[0]][outcome]
            group2 = self.data[self.data[group] == unique_groups[1]][outcome]
            t_stat, p_value = stats.ttest_ind(group1, group2)
        else:
            t_stat, p_value = None, None
        
        return {
            "groups_summary": groups,
            "t_statistic": t_stat,
            "p_value": p_value,
        }
    
    def correlation_analysis(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute correlation matrix.
        
        Args:
            columns (List[str], optional): Columns to correlate
        
        Returns:
            pd.DataFrame: Correlation matrix
        """
        if self.data is None:
            return pd.DataFrame()
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        return self.data[columns].corr()
    
    def check_balance(self, treatment: str, baseline_vars: List[str]) -> Dict:
        """
        Check baseline balance between treatment and control groups.
        
        Args:
            treatment (str): Treatment indicator column
            baseline_vars (List[str]): List of baseline variables
        
        Returns:
            dict: Balance check results
        """
        if self.data is None:
            return {}
        
        results = {}
        treated = self.data[self.data[treatment] == 1]
        control = self.data[self.data[treatment] == 0]
        
        for var in baseline_vars:
            t_stat, p_value = stats.ttest_ind(treated[var], control[var])
            results[var] = {
                "treated_mean": treated[var].mean(),
                "control_mean": control[var].mean(),
                "diff": treated[var].mean() - control[var].mean(),
                "t_stat": t_stat,
                "p_value": p_value,
                "balanced": p_value > 0.05,
            }
        
        return results
