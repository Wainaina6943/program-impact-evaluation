"""
Data Loading Module

Handles loading data from various sources (CSV, Excel, etc.)
and basic data validation.
"""

import pandas as pd
import os
from typing import Union, Dict, Any


class DataLoader:
    """
    Load and manage datasets from various file formats.
    
    Attributes:
        data (pd.DataFrame): The loaded dataset
        file_path (str): Path to the data file
    """
    
    def __init__(self):
        """Initialize the DataLoader."""
        self.data = None
        self.file_path = None
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path (str): Path to CSV file
            **kwargs: Additional arguments to pass to pd.read_csv()
        
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If file does not exist
            pd.errors.ParserError: If file cannot be parsed
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        self.data = pd.read_csv(file_path, **kwargs)
        
        print(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns from {file_path}")
        return self.data
    
    def load_excel(self, file_path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.
        
        Args:
            file_path (str): Path to Excel file
            sheet_name (str): Sheet name to load (default: 0)
            **kwargs: Additional arguments to pass to pd.read_excel()
        
        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If file does not exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        
        print(f"Loaded {len(self.data)} rows and {len(self.data.columns)} columns from {file_path}")
        return self.data
    
    def info(self) -> Dict[str, Any]:
        """
        Get information about loaded data.
        
        Returns:
            dict: Data information including shape, columns, dtypes, missing values
        """
        if self.data is None:
            return {"error": "No data loaded"}
        
        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "dtypes": self.data.dtypes.to_dict(),
            "missing_values": self.data.isnull().sum().to_dict(),
            "memory_usage": self.data.memory_usage(deep=True).sum() / 1024**2,  # MB
        }
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Display first n rows.
        
        Args:
            n (int): Number of rows to display
        
        Returns:
            pd.DataFrame: First n rows
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data.head(n)
    
    def describe(self) -> pd.DataFrame:
        """
        Get summary statistics.
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if self.data is None:
            return pd.DataFrame()
        return self.data.describe()
