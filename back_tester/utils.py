"""
Utility functions for the back tester.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def ensure_directory(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file with proper error handling."""
    try:
        ensure_directory(file_path.parent)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
    except Exception as e:
        raise ValueError(f"Failed to save JSON to {file_path}: {e}")


def load_json(file_path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load data from JSON file with proper error handling."""
    try:
        if not file_path.exists():
            return default or {}
        
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load JSON from {file_path}: {e}")


def append_json(data: Dict[str, Any], file_path: Path) -> None:
    """Append data to JSON file (for transactions)."""
    try:
        ensure_directory(file_path.parent)
        
        # Load existing data
        existing_data = load_json(file_path, default=[])
        
        # Ensure it's a list
        if not isinstance(existing_data, list):
            existing_data = []
        
        # Append new data
        existing_data.append(data)
        
        # Save back
        save_json(existing_data, file_path)
    except Exception as e:
        raise ValueError(f"Failed to append JSON to {file_path}: {e}")


def find_file_in_paths(filename: str, search_paths: List[Path]) -> Optional[Path]:
    """Find a file in a list of search paths."""
    for path in search_paths:
        file_path = path / filename
        if file_path.exists():
            return file_path
    return None


def format_currency(amount: float) -> str:
    """Format amount as currency string."""
    return f"${amount:,.2f}"


def format_percentage(value: float) -> str:
    """Format value as percentage string."""
    return f"{value:.2f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero."""
    try:
        return numerator / denominator if denominator != 0 else default
    except ZeroDivisionError:
        return default


def timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def validate_date_format(date_str: str) -> bool:
    """Validate date string format (YYYY-MM-DD)."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False
