#!/usr/bin/env python3
"""
Optimized radiomics extraction with evidence-based settings for pancreatic CT
Based on IBSI guidelines and pancreas-specific literature (2023-2024)

Key improvements:
- binWidth: 45 HU (optimal for pancreas, was 25)
- resampledPixelSpacing: [2.0, 2.0, 2.0] (was 1.0, reduces over-interpolation)
- Added minimum ROI constraints and numerical stability settings
"""

import numpy as np
import pandas as pd
import nibabel as nib
from radiomics import featureextractor
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import argparse
import os
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths (override via --input-dir/--output-dir or RADPANC_CT_DIR env var)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path(
    os.environ.get(
        "RADPANC_CT_DIR",
        SCRIPT_DIR.parent / "data" / "ct_head_data"
    )
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_optimized"

def get_optimized_settings():
    """
    Get evidence-based optimal settings for pancreatic CT radiomics
    
    References:
    - IBSI Reference Manual v11 (2024)
    - Park et al., "Reproducibility and Generalizability in Radiomics" (2023)
    - Attiyeh et al., Annals of Surgery (2018) - pancreas-specific
    """
    settings = {
        # Binning settings - CRITICAL for variance
        'binWidth': 45,  # Optimal for pancreas (was 25, too narrow)
        
        # Resampling - less aggressive than 1mm
        'resampledPixelSpacing': [2.0, 2.0, 2.0],  # Was [1.0, 1.0, 1.0]
        'interpolator': 'sitkBSpline',
        'padDistance': 10,  # Padding for boundary effects
        
        # HU value preservation
        'normalize': False,  # Preserve absolute HU values
        'normalizeScale': 100,  # If normalization needed later
        'voxelArrayShift': 0,  # No shift - preserve clinical meaning
        
        # ROI constraints
        'minimumROIDimensions': 2,  # At least 2D
        'minimumROISize': 10,  # Minimum 10 voxels
        
        # Numerical stability
        'geometryTolerance': 0.001,  # Was not set
        'correctMask': True,  # Ensure mask validity
        'preCrop': True,  # Reduce memory usage
        
        # GLCM settings
        'distances': [1],  # Standard distance
        'force2D': False,
        'force2Ddimension': 2,
        'symmetricalGLCM': True,
        
        # Label
        'label': 1,
        
        # Settings documentation
        '_comment': 'Optimized settings based on IBSI guidelines and pancreas-specific literature'
    }
    
    return settings

def validate_image_spacing(image_path):
    """Check original image spacing to determine if resampling is needed"""
    img = nib.load(image_path)
    spacing = img.header.get_zooms()[:3]  # Get x, y, z spacing
    
    # If original spacing is already close to 2mm, skip resampling
    if all(1.5 <= s <= 2.5 for s in spacing):
        logger.info(f"  Original spacing {spacing} is optimal, considering no resampling")
        return None  # Signal to use original spacing
    else:
        logger.info(f"  Original spacing {spacing}, will resample to 2mm isotropic")
        return [2.0, 2.0, 2.0]

def extract_optimized_radiomics(input_dir: Path = DEFAULT_INPUT_DIR,
                                output_dir: Path = DEFAULT_OUTPUT_DIR,
                                log_sigma: list[float] | None = None):
    """Extract radiomics with optimized settings"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all patient directories
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(patient_dirs)} patients to process in {input_dir}")
    
    # Get optimized settings
    settings = get_optimized_settings()
    
    logger.info("="*60)
    logger.info("OPTIMIZED RADIOMICS EXTRACTION")
    logger.info("="*60)
    logger.info("Key improvements from previous version:")
    logger.info("  - binWidth: 25 → 45 HU (better for pancreas)")
    logger.info("  - resampling: 1mm → 2mm (less interpolation)")
    logger.info("  - Added ROI constraints and stability settings")
    logger.info("="*60)
    
    # Initialize extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    
    # Enable all feature classes
    extractor.enableAllFeatures()
    
    # Enable image types (Original + filters)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName('Original')
    
    # LoG with configurable sigma values
    if log_sigma is None:
        log_sigma = [2.0, 3.0, 4.0]
    extractor.enableImageTypeByName('LoG', customArgs={'sigma': log_sigma})
    
    # Wavelet decomposition
    extractor.enableImageTypeByName('Wavelet', customArgs={})
    
    logger.info("\nEnabled image types:")
    for imageType, customArgs in extractor.enabledImagetypes.items():
        logger.info(f"  - {imageType}: {customArgs}")
    
    # Process patients
    all_features = []
    failed_patients = []
    
    for patient_dir in tqdm(patient_dirs, desc="Extracting optimized radiomics"):
        patient_name = patient_dir.name
        
        # File paths
        ct_path = patient_dir / "ct_head.nii.gz"
        mask_path = patient_dir / "head_mask_cropped.nii.gz"
        
        if not ct_path.exists() or not mask_path.exists():
            logger.warning(f"Missing files for {patient_name}")
            failed_patients.append(patient_name)
            continue
        
        try:
            # Check if we should use original spacing
            optimal_spacing = validate_image_spacing(ct_path)
            
            # Update settings if needed
            if optimal_spacing is None:
                # Use original spacing
                current_settings = settings.copy()
                current_settings['resampledPixelSpacing'] = None
                current_extractor = featureextractor.RadiomicsFeatureExtractor(**current_settings)
                current_extractor.enableAllFeatures()
                current_extractor.disableAllImageTypes()
                current_extractor.enableImageTypeByName('Original')
                current_extractor.enableImageTypeByName('LoG', customArgs={'sigma': log_sigma})
                current_extractor.enableImageTypeByName('Wavelet', customArgs={})
            else:
                current_extractor = extractor
            
            # Extract features
            features = current_extractor.execute(str(ct_path), str(mask_path))
            
            # Create feature dictionary
            feature_dict = {'patient_id': patient_name}
            
            # Add all non-diagnostic features
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    # Convert numpy types
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        value = float(value.item())
                    feature_dict[key] = value
            
            # Add extraction metadata
            feature_dict['extraction_settings'] = 'optimized_v1'
            feature_dict['binWidth_used'] = settings['binWidth']
            feature_dict['resampling_used'] = optimal_spacing if optimal_spacing else 'original'
            
            all_features.append(feature_dict)
            
            # Log progress every 10 patients
            if len(all_features) % 10 == 0:
                # Count features by type
                original = len([k for k in feature_dict.keys() if 'original_' in k])
                log = len([k for k in feature_dict.keys() if 'log-sigma-' in k])
                wavelet = len([k for k in feature_dict.keys() if 'wavelet-' in k])
                logger.info(f"Progress: {len(all_features)} patients. Features: Original={original}, LoG={log}, Wavelet={wavelet}")
        
        except Exception as e:
            logger.error(f"Error processing {patient_name}: {str(e)}")
            failed_patients.append(patient_name)
            continue
    
    # Create DataFrame
    if all_features:
        df = pd.DataFrame(all_features)
        
        # Calculate variance statistics for comparison
        feature_cols = [c for c in df.columns if c not in ['patient_id', 'extraction_settings', 
                                                            'binWidth_used', 'resampling_used']]
        
        # Calculate coefficient of variation for each feature
        feature_means = df[feature_cols].mean()
        feature_stds = df[feature_cols].std()
        feature_cvs = (feature_stds / feature_means.abs()).replace([np.inf, -np.inf], np.nan)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"radiomics_optimized_{timestamp}.csv"
        df.to_csv(output_path, index=False)
        
        # Feature summary
        original_cols = [c for c in feature_cols if 'original_' in c]
        log_cols = [c for c in feature_cols if 'log-sigma-' in c]
        wavelet_cols = [c for c in feature_cols if 'wavelet-' in c]
        
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION COMPLETE - OPTIMIZED SETTINGS")
        logger.info("="*60)
        logger.info(f"Patients processed: {len(df)}")
        logger.info(f"Failed patients: {len(failed_patients)}")
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"  - Original: {len(original_cols)}")
        logger.info(f"  - LoG: {len(log_cols)}")
        logger.info(f"  - Wavelet: {len(wavelet_cols)}")
        logger.info("\nVariance Statistics:")
        logger.info(f"  - Mean CV across features: {feature_cvs.mean():.3f}")
        logger.info(f"  - Median CV: {feature_cvs.median():.3f}")
        logger.info(f"  - Features with CV > 0.5: {(feature_cvs > 0.5).sum()}")
        logger.info(f"  - Features with CV > 1.0: {(feature_cvs > 1.0).sum()}")
        logger.info(f"\nOutput saved to: {output_path}")
        logger.info("="*60)

        # Create detailed report
        report = {
            'timestamp': timestamp,
            'extraction_version': 'optimized_v1',
            'patients_processed': len(df),
            'failed_patients': failed_patients,
            'total_features': len(feature_cols),
            'feature_breakdown': {
                'original': len(original_cols),
                'log': len(log_cols),
                'wavelet': len(wavelet_cols)
            },
            'settings_used': settings,
            'log_sigma_used': log_sigma,
            'variance_statistics': {
                'mean_cv': float(feature_cvs.mean()),
                'median_cv': float(feature_cvs.median()),
                'high_variance_features': int((feature_cvs > 0.5).sum()),
                'very_high_variance_features': int((feature_cvs > 1.0).sum())
            }
        }
        report_path = output_dir / f"extraction_report_optimized_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Detailed report saved to: {report_path}")

        return df
    else:
        logger.error("No features extracted!")
        return None


def parse_args():
    parser = argparse.ArgumentParser(description="Optimized radiomics extraction (pancreatic head)")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="Directory containing per-patient ct_head.nii.gz and head_mask_cropped.nii.gz")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write feature CSVs")
    parser.add_argument(
        "--log-sigma",
        type=str,
        default="2,3,4",
        help="Comma-separated LoG sigma list (e.g. '3,5,7')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sigma_vals = [float(x) for x in args.log_sigma.split(",") if x.strip()]
    extract_optimized_radiomics(args.input_dir, args.output_dir, log_sigma=sigma_vals)
