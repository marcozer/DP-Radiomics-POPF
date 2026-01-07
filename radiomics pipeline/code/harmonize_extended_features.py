#!/usr/bin/env python3
"""
Apply ComBat harmonization to extended radiomics features
Fixed version that handles missing scanner metadata gracefully
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import argparse
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (override via CLI)
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PIPELINE_DIR / "outputs"
EXTENDED_DIR = PIPELINE_DIR / "outputs_extended_fixed"
SCANNER_METADATA_PATH = OUTPUT_DIR / "scanner_metadata.csv"

def find_scanner_metadata(scanner_metadata_path: Path | None = None):
    """Find scanner metadata file using provided path or common defaults."""
    candidates = []
    if scanner_metadata_path:
        candidates.append(Path(scanner_metadata_path))
    candidates.extend([
        SCANNER_METADATA_PATH,
        OUTPUT_DIR / "scanner_metadata_complete.csv",
    ])
    for path in candidates:
        if path.exists():
            logger.info(f"Found scanner metadata at: {path}")
            return path
    return None

def extract_scanner_from_dicom():
    """Quick extraction of scanner info if metadata is missing"""
    logger.info("Attempting to extract scanner information from existing metadata...")
    
    # Check if we already have scanner metadata in outputs
    scanner_files = list(OUTPUT_DIR.glob("scanner_metadata*.csv"))
    if scanner_files:
        latest = max(scanner_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found existing scanner metadata: {latest}")
        return latest
    
    # If not, we need to extract it
    logger.warning("No scanner metadata found. You need to run extract_scanner_metadata.py first.")
    return None

def load_and_prepare_data(features_path: Path | None = None,
                          scanner_metadata_path: Path | None = None):
    """Load radiomics data and scanner metadata."""
    # Resolve features file
    if features_path:
        features_path = Path(features_path)
        if not features_path.exists():
            logger.error(f"Features file not found: {features_path}")
            return None, None
        logger.info(f"Loading features from: {features_path}")
        features_df = pd.read_csv(features_path)
    elif EXTENDED_DIR.exists():
        extended_files = list(EXTENDED_DIR.glob("radiomics_extended_fixed_*.csv"))
        if extended_files:
            features_path = max(extended_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading extended features from: {features_path}")
            features_df = pd.read_csv(features_path)
        else:
            logger.error("No extended features files found!")
            return None, None
    else:
        logger.warning("Extended features directory not found. Using standard features...")
        standard_files = list(OUTPUT_DIR.glob("radiomics_features_2*.csv"))
        non_harmonized = [f for f in standard_files if "harmonized" not in f.name]
        if non_harmonized:
            features_path = max(non_harmonized, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading standard features from: {features_path}")
            features_df = pd.read_csv(features_path)
        else:
            logger.error("No radiomics files found!")
            return None, None
    
    # Load scanner metadata
    scanner_path = find_scanner_metadata(scanner_metadata_path)
    if scanner_path and scanner_path.exists():
        scanner_df = pd.read_csv(scanner_path)
        logger.info(f"Loaded scanner metadata: {len(scanner_df)} entries")
        return features_df, scanner_df
    else:
        logger.warning("Scanner metadata not found. Checking if all patients use same scanner...")
        return features_df, None

def harmonize_features(features_df, scanner_df=None):
    """Apply harmonization if multiple scanners present"""
    
    if scanner_df is None:
        logger.warning("No scanner metadata available. Skipping harmonization.")
        logger.info("If your data is from multiple scanners, run extract_scanner_metadata.py first.")
        return features_df
    
    # Merge scanner info with features
    patient_col = 'patient_id' if 'patient_id' in features_df.columns else 'PatientID'
    scanner_patient_col = 'PatientName' if 'PatientName' in scanner_df.columns else 'patient_name'
    
    # Create scanner ID
    scanner_df['scanner_id'] = scanner_df['Manufacturer'] + "_" + scanner_df['ManufacturerModelName']
    
    # Merge
    merged_df = pd.merge(
        features_df,
        scanner_df[['PatientName', 'scanner_id', 'Manufacturer', 'ManufacturerModelName']],
        left_on=patient_col,
        right_on='PatientName',
        how='left'
    )
    
    # Check scanner distribution
    scanner_counts = merged_df['scanner_id'].value_counts()
    logger.info(f"Scanner distribution:\n{scanner_counts}")
    
    if len(scanner_counts) == 1:
        logger.info("All patients from same scanner. No harmonization needed.")
        return features_df
    
    # Apply ComBat harmonization
    try:
        from neuroCombat import neuroCombat
        
        # Prepare features for harmonization
        feature_cols = [col for col in features_df.columns 
                       if col not in [patient_col, 'extraction_timestamp', 'head_volume_ml', 'mean_hu']
                       and not col.startswith('shape_')]  # Exclude shape features
        
        # Separate different feature types
        original_cols = [c for c in feature_cols if 'original_' in c or 
                        (not 'log-sigma-' in c and not 'wavelet-' in c)]
        log_cols = [c for c in feature_cols if 'log-sigma-' in c]
        wavelet_cols = [c for c in feature_cols if 'wavelet-' in c]
        
        logger.info(f"Features to harmonize:")
        logger.info(f"  - Original: {len(original_cols)}")
        logger.info(f"  - LoG: {len(log_cols)}")
        logger.info(f"  - Wavelet: {len(wavelet_cols)}")
        
        # Filter to valid scanners (>= 3 patients)
        valid_scanners = scanner_counts[scanner_counts >= 3].index
        valid_mask = merged_df['scanner_id'].isin(valid_scanners)
        
        if len(valid_scanners) < 2:
            logger.warning("Not enough scanner groups for harmonization (need at least 2 with >=3 patients each)")
            return features_df
        
        # Apply harmonization
        logger.info(f"Applying ComBat to {len(feature_cols)} features...")
        
        # Prepare data
        features_to_harmonize = merged_df.loc[valid_mask, feature_cols].T.values
        batch_labels = merged_df.loc[valid_mask, 'scanner_id'].values
        
        # Encode batches
        batch_dict = {scanner: i for i, scanner in enumerate(np.unique(batch_labels))}
        batch_encoded = np.array([batch_dict[b] for b in batch_labels])
        
        # Create covariates DataFrame as required by neuroCombat
        covars_df = pd.DataFrame({'batch': batch_encoded})
        
        # Apply ComBat
        harmonized_data = neuroCombat(
            dat=features_to_harmonize,
            covars=covars_df,
            batch_col='batch'
        )['data']
        
        # Create harmonized dataframe
        harmonized_df = features_df.copy()
        harmonized_df.loc[valid_mask, feature_cols] = harmonized_data.T
        
        # Add harmonization flag
        harmonized_df['harmonized'] = False
        harmonized_df.loc[valid_mask, 'harmonized'] = True
        
        logger.info(f"Harmonization complete. {valid_mask.sum()} patients harmonized.")
        
        return harmonized_df
        
    except ImportError:
        logger.error("neuroCombat not found. Install it from: https://github.com/Jfortin1/neuroCombat_Python")
        return features_df
    except Exception as e:
        logger.error(f"Harmonization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return features_df

def main():
    """Main harmonization pipeline for extended features"""
    logger.info("="*60)
    logger.info("EXTENDED FEATURES HARMONIZATION")
    logger.info("="*60)

    parser = argparse.ArgumentParser(description="Harmonize radiomics features with ComBat")
    parser.add_argument("--features-path", type=Path, default=None,
                        help="Explicit path to features CSV (overrides auto-discovery)")
    parser.add_argument("--scanner-metadata", type=Path, default=None,
                        help="Path to scanner metadata CSV")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory where harmonized outputs are written")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    features_df, scanner_df = load_and_prepare_data(
        features_path=args.features_path,
        scanner_metadata_path=args.scanner_metadata
    )
    
    if features_df is None:
        logger.error("Failed to load radiomics features!")
        return
    
    logger.info(f"Loaded features: {features_df.shape}")
    
    # Count feature types
    feature_cols = [c for c in features_df.columns if c not in ['patient_id', 'PatientID', 'extraction_timestamp']]
    original = len([c for c in feature_cols if 'original_' in c or 
                   (not 'log-sigma-' in c and not 'wavelet-' in c and 'shape_' not in c)])
    log = len([c for c in feature_cols if 'log-sigma-' in c])
    wavelet = len([c for c in feature_cols if 'wavelet-' in c])
    
    logger.info(f"Feature breakdown: Original={original}, LoG={log}, Wavelet={wavelet}")
    
    # Apply harmonization
    harmonized_df = harmonize_features(features_df, scanner_df)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if EXTENDED_DIR.exists():
        output_path = EXTENDED_DIR / f"radiomics_extended_harmonized_{timestamp}.csv"
    else:
        output_path = out_dir / f"radiomics_harmonized_{timestamp}.csv"
    
    harmonized_df.to_csv(output_path, index=False)
    logger.info(f"Saved harmonized features to: {output_path}")
    
    # Generate report
    report = {
        'timestamp': timestamp,
        'input_shape': list(features_df.shape),
        'output_shape': list(harmonized_df.shape),
        'feature_types': {
            'original': original,
            'log': log,
            'wavelet': wavelet
        },
        'harmonization_applied': 'harmonized' in harmonized_df.columns
    }
    
    report_path = output_path.parent / f"harmonization_report_{timestamp}.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("="*60)
    logger.info("HARMONIZATION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()
