"""
Impact Evaluation Module

Implements various impact evaluation methodologies:
- Randomized Controlled Trials (RCT)
- Difference-in-Differences (DiD)
- Propensity Score Matching (PSM)
- Regression Discontinuity Design (RDD)
- Instrumental Variables (IV)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import NearestNeighbors


class ImpactEvaluator:
    """
    Evaluate program impact using various methodologies.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the ImpactEvaluator.
        
        Args:
            data (pd.DataFrame, optional): Dataset for evaluation
        """
        self.data = data
        self.results = {}
    
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the data for evaluation.
        
        Args:
            data (pd.DataFrame): Dataset for evaluation
        """
        self.data = data
    
    def rct_analysis(self, outcome: str, treatment: str, **kwargs) -> Dict:
        """
        Analyze impact using Randomized Controlled Trial (RCT) design.
        
        Compare mean outcomes between treatment and control groups.
        
        Args:
            outcome (str): Outcome variable column name
            treatment (str): Treatment indicator column name
            **kwargs: Additional arguments
        
        Returns:
            dict: RCT results including treatment effect, p-value, confidence interval
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        treated = self.data[self.data[treatment] == 1][outcome]
        control = self.data[self.data[treatment] == 0][outcome]
        
        # Calculate treatment effect
        ate = treated.mean() - control.mean()
        
        # T-test
        t_stat, p_value = stats.ttest_ind(treated, control)
        
        # Confidence interval
        se = np.sqrt(treated.var() / len(treated) + control.var() / len(control))
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        results = {
            "method": "RCT",
            "treatment_effect": ate,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "treated_n": len(treated),
            "control_n": len(control),
            "treated_mean": treated.mean(),
            "control_mean": control.mean(),
        }
        
        self.results["rct"] = results
        return results
    
    def did_analysis(self, outcome: str, treatment: str, time: str, **kwargs) -> Dict:
        """
        Difference-in-Differences (DiD) analysis.
        
        Estimates treatment effect using before-after and treatment-control comparisons.
        
        Args:
            outcome (str): Outcome variable column name
            treatment (str): Treatment indicator column name
            time (str): Time period indicator column name
            **kwargs: Additional arguments
        
        Returns:
            dict: DiD results
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        # Prepare data
        data = self.data.copy()
        data["interaction"] = data[treatment] * data[time]
        
        # OLS regression: outcome = β0 + β1*treatment + β2*time + β3*interaction
        X = data[[treatment, time, "interaction"]]
        X = sm.add_constant(X)
        y = data[outcome]
        
        model = sm.OLS(y, X).fit()
        
        results = {
            "method": "Difference-in-Differences",
            "treatment_effect": model.params["interaction"],
            "standard_error": model.bse["interaction"],
            "t_statistic": model.tvalues["interaction"],
            "p_value": model.pvalues["interaction"],
            "ci_lower": model.conf_int(0.05).loc["interaction", 0],
            "ci_upper": model.conf_int(0.05).loc["interaction", 1],
            "r_squared": model.rsquared,
            "nobs": model.nobs,
        }
        
        self.results["did"] = results
        return results
    
    def psm_analysis(self, outcome: str, treatment: str, covariates: List[str], **kwargs) -> Dict:
        """
        Propensity Score Matching (PSM) analysis.
        
        Match treated and control units based on propensity scores.
        
        Args:
            outcome (str): Outcome variable column name
            treatment (str): Treatment indicator column name
            covariates (List[str]): List of covariate columns
            **kwargs: Additional arguments
        
        Returns:
            dict: PSM results
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        data = self.data.copy()
        
        # Estimate propensity scores
        X = data[covariates]
        X = sm.add_constant(X)
        y = data[treatment]
        
        ps_model = sm.Logit(y, X).fit(disp=0)
        data["propensity_score"] = ps_model.predict(X)
        
        # Match on propensity score
        treated_idx = data[data[treatment] == 1].index
        control_idx = data[data[treatment] == 0].index
        
        treated_ps = data.loc[treated_idx, "propensity_score"].values.reshape(-1, 1)
        control_ps = data.loc[control_idx, "propensity_score"].values.reshape(-1, 1)
        
        # Simple nearest neighbor matching
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(control_ps)
        distances, indices = nn.kneighbors(treated_ps)
        
        matched_treated = data.loc[treated_idx, outcome].values
        matched_control = data.loc[control_idx[indices.flatten()], outcome].values
        
        ate = matched_treated.mean() - matched_control.mean()
        se = np.sqrt(matched_treated.var() / len(matched_treated) + 
                     matched_control.var() / len(matched_control))
        t_stat = ate / se
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(matched_treated) - 1))
        
        results = {
            "method": "Propensity Score Matching",
            "treatment_effect": ate,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_value,
            "matched_pairs": len(matched_treated),
            "mean_propensity_score_difference": distances.mean(),
        }
        
        self.results["psm"] = results
        return results
    
    def get_results_summary(self) -> Dict:
        """
        Get summary of all evaluation results.
        
        Returns:
            dict: All evaluation results
        """
        return self.results
