"""
Visualization Module

Create plots and visualizations for impact evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict


class Visualizer:
    """
    Create visualizations for impact evaluation analysis.
    """
    
    def __init__(self, style: str = "seaborn-v0_8-darkgrid"):
        """
        Initialize the Visualizer.
        
        Args:
            style (str): Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            pass  # Use default style if specified style not available
        
        self.fig_size = (12, 6)
    
    def plot_treatment_effect(self, results: Dict, title: str = "Treatment Effect") -> None:
        """
        Plot treatment effect with confidence interval.
        
        Args:
            results (dict): Results dictionary with 'treatment_effect', 'ci_lower', 'ci_upper'
            title (str): Plot title
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        effect = results.get("treatment_effect", 0)
        ci_lower = results.get("ci_lower", 0)
        ci_upper = results.get("ci_upper", 0)
        
        ax.errorbar(1, effect, yerr=[[effect - ci_lower], [ci_upper - effect]], 
                   fmt="o", markersize=10, capsize=10, capthick=2)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 2)
        ax.set_ylabel("Treatment Effect")
        ax.set_title(title)
        ax.set_xticks([])
        
        # Add text annotation
        ax.text(1, effect, f"ATE: {effect:.3f}\n95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]",
               horizontalalignment="center", verticalalignment="bottom")
        
        plt.tight_layout()
        plt.show()
    
    def plot_group_comparison(self, data: pd.DataFrame, outcome: str, group: str) -> None:
        """
        Create box plot comparing outcomes across groups.
        
        Args:
            data (pd.DataFrame): Data to plot
            outcome (str): Outcome variable column name
            group (str): Grouping variable column name
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        sns.boxplot(data=data, x=group, y=outcome, ax=ax)
        ax.set_title(f"{outcome} by {group}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_correlation_heatmap(self, data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Create correlation heatmap.
        
        Args:
            data (pd.DataFrame): Data to analyze
            columns (List[str], optional): Columns to include in heatmap
        """
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        corr_matrix = data[columns].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                   center=0, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Correlation Matrix")
        
        plt.tight_layout()
        plt.show()
    
    def plot_distribution(self, data: pd.DataFrame, column: str, bins: int = 30) -> None:
        """
        Create histogram with KDE.
        
        Args:
            data (pd.DataFrame): Data to plot
            column (str): Column to visualize
            bins (int): Number of bins for histogram
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        data[column].hist(bins=bins, ax=ax, alpha=0.7, edgecolor="black")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {column}")
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def set_style(style: str = "darkgrid", palette: str = "husl") -> None:
        """
        Set seaborn style and palette.
        
        Args:
            style (str): Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
            palette (str): Color palette name
        """
        sns.set_style(style)
        sns.set_palette(palette)
