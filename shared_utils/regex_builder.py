"""
Regex rules builder module for combining component files into master regex rules.

This module provides functionality to build regex_rules_master.json from component files
and internal patterns. The master file should be used as the only source for static scanning.
"""

import json
import pathlib
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def build_regex_rules_master(data_dir: Optional[pathlib.Path] = None) -> pathlib.Path:
    """
    Build regex_rules_master.json by combining component files with regex_internal.json.
    
    Args:
        data_dir: Directory containing component files. Defaults to data/sente/static_hits.
        
    Returns:
        Path to the generated master file.
        
    This function should be used as the only source for static scanning.
    """
    if data_dir is None:
        data_dir = pathlib.Path("data/sente/static_hits")
    
    master_rules = {
        "_meta": {
            "version": "0.1.0",
            "spdx_license": "Apache-2.0",
            "notes": "Auto-generated master regex rules from component files and internal patterns.",
            "provenance": [
                "Combined from component files: secrets.json, pii.json, jailbreaks.json, exfil.json, injections.json, cloud.json",
                "Plus internal patterns from regex_internal.json"
            ]
        }
    }
    
    # Component files to combine
    component_files = [
        "secrets.json",
        "pii.json", 
        "jailbreaks.json",
        "exfil.json",
        "injections.json",
        "cloud.json"
    ]
    
    total_component_rules = 0
    
    # Load and combine component files
    for filename in component_files:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    component_data = json.load(f)
                
                # Add rules from component file (skip _meta)
                component_rule_count = 0
                for pattern, rule_data in component_data.items():
                    if pattern != "_meta":
                        master_rules[pattern] = rule_data
                        component_rule_count += 1
                        
                total_component_rules += component_rule_count
                logger.info(f"Loaded {component_rule_count} rules from {filename}")
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
        else:
            logger.warning(f"Component file not found: {filename}")
    
    # Load and combine internal patterns
    internal_filepath = data_dir / "regex_internal.json"
    internal_rule_count = 0
    if internal_filepath.exists():
        try:
            with open(internal_filepath, "r") as f:
                internal_data = json.load(f)
            
            # Add internal patterns
            for pattern, rule_data in internal_data.items():
                master_rules[pattern] = rule_data
                internal_rule_count += 1
                
            logger.info(f"Loaded {internal_rule_count} internal patterns")
        except Exception as e:
            logger.warning(f"Error loading regex_internal.json: {e}")
    else:
        logger.warning("Internal patterns file not found: regex_internal.json")
    
    # Save the master file
    master_filepath = data_dir / "regex_rules_master.json"
    with open(master_filepath, "w") as f:
        json.dump(master_rules, f, indent=2)
    
    total_rules = len(master_rules) - 1  # Subtract 1 for _meta
    logger.info(f"Generated regex_rules_master.json with {total_rules} total rules")
    logger.info(f"  - {total_component_rules} component rules")
    logger.info(f"  - {internal_rule_count} internal rules")
    
    return master_filepath

def get_component_file_paths(data_dir: Optional[pathlib.Path] = None) -> Dict[str, pathlib.Path]:
    """
    Get paths to all component regex rule files.
    
    Args:
        data_dir: Directory containing component files. Defaults to data/sente/static_hits.
        
    Returns:
        Dictionary mapping component names to their file paths.
    """
    if data_dir is None:
        data_dir = pathlib.Path("data/sente/static_hits")
    
    component_files = {
        "secrets": data_dir / "secrets.json",
        "pii": data_dir / "pii.json",
        "jailbreaks": data_dir / "jailbreaks.json",
        "exfil": data_dir / "exfil.json",
        "injections": data_dir / "injections.json",
        "cloud": data_dir / "cloud.json",
        "internal": data_dir / "regex_internal.json"
    }
    
    return component_files

def validate_component_files(data_dir: Optional[pathlib.Path] = None) -> Dict[str, bool]:
    """
    Validate that all component files exist and are valid JSON.
    
    Args:
        data_dir: Directory containing component files. Defaults to data/sente/static_hits.
        
    Returns:
        Dictionary mapping component names to validation status.
    """
    if data_dir is None:
        data_dir = pathlib.Path("data/sente/static_hits")
    
    component_paths = get_component_file_paths(data_dir)
    validation_results = {}
    
    for name, filepath in component_paths.items():
        if not filepath.exists():
            validation_results[name] = False
            logger.warning(f"Component file missing: {name} ({filepath})")
            continue
            
        try:
            with open(filepath, "r") as f:
                json.load(f)
            validation_results[name] = True
            logger.info(f"Component file valid: {name}")
        except json.JSONDecodeError as e:
            validation_results[name] = False
            logger.error(f"Component file invalid JSON: {name} - {e}")
        except Exception as e:
            validation_results[name] = False
            logger.error(f"Error reading component file: {name} - {e}")
    
    return validation_results

def get_rule_statistics(data_dir: Optional[pathlib.Path] = None) -> Dict[str, int]:
    """
    Get statistics about rules in each component file.
    
    Args:
        data_dir: Directory containing component files. Defaults to data/sente/static_hits.
        
    Returns:
        Dictionary mapping component names to rule counts.
    """
    if data_dir is None:
        data_dir = pathlib.Path("data/sente/static_hits")
    
    component_paths = get_component_file_paths(data_dir)
    statistics = {}
    
    for name, filepath in component_paths.items():
        if not filepath.exists():
            statistics[name] = 0
            continue
            
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            # Count rules (exclude _meta)
            rule_count = len([k for k in data.keys() if k != "_meta"])
            statistics[name] = rule_count
        except Exception as e:
            logger.warning(f"Error counting rules in {name}: {e}")
            statistics[name] = 0
    
    return statistics
