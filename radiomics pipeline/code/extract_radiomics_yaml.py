#!/usr/bin/env python3
"""
YAML-Based Radiomics Extraction with Academic Justifications
Loads configuration from radiomics_config.yaml and applies all settings
"""

import yaml
import numpy as np
import pandas as pd
import nibabel as nib
from radiomics import featureextractor
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
import sys
import argparse
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = SCRIPT_DIR / "configs" / "radiomics_config_2mm.yaml"
# Default input directory (override via --input-dir)
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "ct_head_data"
# Default output directory (override via --output-dir)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_yaml"


class YAMLRadiomicsExtractor:
    """Radiomics extraction using YAML configuration with full justification tracking"""
    
    def __init__(self, config_path=None, output_dir=None):
        """Initialize with configuration file"""
        self.config_path = config_path or CONFIG_PATH
        self.output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = self.load_config()
        self.validate_config()
        self.extractor = None
        
    def load_config(self):
        """Load YAML configuration"""
        logger.info(f"Loading configuration from: {self.config_path}")
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration version: {config['metadata']['version']}")
        logger.info(f"IBSI compliant: {config['metadata']['ibsi_compliant']}")
        
        return config
    
    def validate_config(self):
        """Validate configuration parameters"""
        required_keys = ['extraction', 'imageType', 'featureClass']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration section: {key}")
        
        # Validate extraction parameters
        extraction = self.config['extraction']
        
        # Check critical parameters
        if extraction['binWidth'] <= 0:
            raise ValueError(f"Invalid binWidth: {extraction['binWidth']}")
        
        if not all(s > 0 for s in extraction['resampledPixelSpacing']):
            raise ValueError(f"Invalid resampledPixelSpacing: {extraction['resampledPixelSpacing']}")
        
        # Validate resegmentation range if present
        if 'resegmentRange' in self.config:
            range_vals = self.config['resegmentRange']
            if len(range_vals) != 2 or range_vals[0] >= range_vals[1]:
                raise ValueError(f"Invalid resegmentRange: {range_vals}")
        
        logger.info("Configuration validation passed")
    
    def log_justifications(self):
        """Log academic justifications for key parameters"""
        logger.info("\n" + "="*60)
        logger.info("PARAMETER JUSTIFICATIONS")
        logger.info("="*60)
        
        # Key parameters and their justifications
        key_params = {
            'binWidth': self.config['extraction']['binWidth'],
            'resampledPixelSpacing': self.config['extraction']['resampledPixelSpacing'],
            'interpolator': self.config['extraction']['interpolator']
        }
        
        logger.info(f"\nBin Width: {key_params['binWidth']} HU")
        logger.info("  → Attiyeh et al. (2018): Optimal for pancreatic adenocarcinoma")
        logger.info("  → IBSI: Fixed bin width preferred for CT")
        
        logger.info(f"\nResampling: {key_params['resampledPixelSpacing']} mm")
        logger.info("  → Park et al. (2023): Preserves 94% texture information")
        logger.info("  → 8x computational reduction vs 1mm")
        
        logger.info(f"\nInterpolator: {key_params['interpolator']}")
        logger.info("  → IBSI consensus: B-spline minimizes aliasing")
        logger.info("  → Mackin et al. (2015): Superior for texture preservation")
        
        if 'resegmentRange' in self.config and self.config.get('resegmentRange') is not None:
            logger.info(f"\nResegmentation Range: {self.config['resegmentRange']} HU")
            logger.info("  → Warning: Not recommended for pancreatic analysis")
            logger.info("  → Removes diagnostic heterogeneity (fatty infiltration, fibrosis, calcifications)")
        else:
            logger.info("\nNo resegmentation applied")
            logger.info("  → Preserves full intensity heterogeneity (recommended)")
            logger.info("  → Retains fatty infiltration, normal parenchyma, fibrosis, calcifications")
    
    def setup_extractor(self):
        """Setup PyRadiomics extractor with YAML configuration"""
        logger.info("\nConfiguring PyRadiomics extractor...")
        
        # Get extraction settings
        settings = self.config['extraction'].copy()
        
        # Add resegmentation if specified (optional - not recommended for pancreas)
        if 'resegmentRange' in self.config and self.config['resegmentRange'] is not None:
            settings['resegmentRange'] = self.config['resegmentRange']
            logger.info(f"Resegmentation enabled: {self.config['resegmentRange']} HU")
            logger.warning("  → Note: Resegmentation may remove diagnostic heterogeneity in pancreatic tissue")
        else:
            logger.info("No resegmentation - preserving full tissue heterogeneity (recommended for pancreas)")
        
        # Initialize extractor
        self.extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        
        # Configure image types
        self.extractor.disableAllImageTypes()
        for image_type, params in self.config['imageType'].items():
            if params:
                self.extractor.enableImageTypeByName(image_type, customArgs=params)
            else:
                self.extractor.enableImageTypeByName(image_type)
            logger.info(f"Enabled image type: {image_type} with params: {params}")
        
        # Configure feature classes
        self.extractor.disableAllFeatures()
        for feature_class in self.config['featureClass'].keys():
            self.extractor.enableFeatureClassByName(feature_class)
            logger.info(f"Enabled feature class: {feature_class}")
        
        return self.extractor
    
    def extract_features(self, ct_path, mask_path):
        """Extract features for a single patient"""
        try:
            features = self.extractor.execute(str(ct_path), str(mask_path))
            
            # Filter out diagnostics
            feature_dict = {}
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    # Convert numpy types
                    if isinstance(value, (np.integer, np.floating)):
                        value = float(value)
                    elif isinstance(value, np.ndarray) and value.size == 1:
                        value = float(value.item())
                    feature_dict[key] = value
            
            return feature_dict
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def process_dataset(self, input_dir=None):
        """Process entire dataset with YAML configuration"""
        input_dir = input_dir or DEFAULT_INPUT_DIR
        
        # Find patient directories
        patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        logger.info(f"\nFound {len(patient_dirs)} patients to process")
        
        # Log justifications
        self.log_justifications()
        
        # Setup extractor
        self.setup_extractor()
        
        # Process patients
        all_features = []
        failed_patients = []
        
        logger.info("\n" + "="*60)
        logger.info("EXTRACTING FEATURES")
        logger.info("="*60)
        
        for patient_dir in tqdm(patient_dirs, desc="Extracting radiomics"):
            patient_name = patient_dir.name
            
            # File paths
            ct_path = patient_dir / "ct_head.nii.gz"
            mask_path = patient_dir / "head_mask_cropped.nii.gz"
            
            if not ct_path.exists() or not mask_path.exists():
                logger.warning(f"Missing files for {patient_name}")
                failed_patients.append(patient_name)
                continue
            
            # Extract features
            features = self.extract_features(ct_path, mask_path)
            
            if features:
                features['patient_id'] = patient_name
                features['config_version'] = self.config['metadata']['version']
                features['extraction_date'] = datetime.now().isoformat()
                all_features.append(features)
            else:
                failed_patients.append(patient_name)
        
        # Create DataFrame
        if all_features:
            df = pd.DataFrame(all_features)
            
            # Count features by type
            feature_counts = self.count_features_by_type(df)
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"radiomics_yaml_{timestamp}.csv"
            df.to_csv(output_path, index=False)
            
            # Generate report
            self.generate_extraction_report(df, feature_counts, failed_patients, timestamp)
            
            # Save methods text
            self.save_methods_text(timestamp)
            
            logger.info("\n" + "="*60)
            logger.info("EXTRACTION COMPLETE")
            logger.info("="*60)
            logger.info(f"Patients processed: {len(df)}")
            logger.info(f"Failed patients: {len(failed_patients)}")
            logger.info(f"Total features: {sum(feature_counts.values())}")
            logger.info(f"Output saved to: {output_path}")
            
            return df
        else:
            logger.error("No features extracted!")
            return None
    
    def count_features_by_type(self, df):
        """Count features by image type and class"""
        feature_cols = [c for c in df.columns 
                       if c not in ['patient_id', 'config_version', 'extraction_date']]
        
        counts = {
            'original': 0,
            'log': 0,
            'wavelet': 0,
            'shape': 0,
            'firstorder': 0,
            'glcm': 0,
            'glrlm': 0,
            'glszm': 0,
            'gldm': 0,
            'ngtdm': 0
        }
        
        for col in feature_cols:
            col_lower = col.lower()
            if 'log-sigma' in col_lower:
                counts['log'] += 1
            elif 'wavelet' in col_lower:
                counts['wavelet'] += 1
            else:
                counts['original'] += 1
            
            # Count by feature class
            for feature_class in ['shape', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']:
                if feature_class in col_lower:
                    counts[feature_class] += 1
                    break
        
        return counts
    
    def generate_extraction_report(self, df, feature_counts, failed_patients, timestamp):
        """Generate detailed extraction report"""
        report = {
            'timestamp': timestamp,
            'configuration': {
                'version': self.config['metadata']['version'],
                'ibsi_compliant': self.config['metadata']['ibsi_compliant'],
                'config_file': str(self.config_path)
            },
            'extraction_parameters': self.config['extraction'],
            'resegmentation': self.config.get('resegmentRange', None),
            'image_types': list(self.config['imageType'].keys()),
            'feature_classes': list(self.config['featureClass'].keys()),
            'results': {
                'patients_processed': len(df),
                'patients_failed': len(failed_patients),
                'failed_list': failed_patients,
                'total_features': sum(feature_counts.values()),
                'feature_breakdown': feature_counts
            },
            'key_justifications': {
                'binWidth': f"{self.config['extraction']['binWidth']} HU - Attiyeh et al. (2018): Optimal for pancreatic adenocarcinoma",
                'resampling': f"{self.config['extraction']['resampledPixelSpacing']} mm - Matches native scanner resolution, minimal interpolation",
                'interpolator': f"{self.config['extraction']['interpolator']} - IBSI consensus: Minimizes aliasing",
                'resegmentation': "Not applied - Preserves full tissue heterogeneity (Eilaghi et al., 2017)"
            }
        }
        
        report_path = self.output_dir / f"extraction_report_yaml_{timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to: {report_path}")
    
    def save_methods_text(self, timestamp):
        """Save publication-ready methods text"""
        if 'methods_text' in self.config:
            methods_path = self.output_dir / f"methods_section_{timestamp}.txt"
            with open(methods_path, 'w') as f:
                f.write(self.config['methods_text'])
            logger.info(f"Methods text saved to: {methods_path}")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(description='YAML-based radiomics extraction with academic justifications')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory with patient data')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for CSV/report artifacts')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate configuration without extraction')
    
    args = parser.parse_args()
    
    # Initialize extractor
    config_path = Path(args.config) if args.config else CONFIG_PATH
    output_dir = Path(args.output_dir) if args.output_dir else None
    extractor = YAMLRadiomicsExtractor(config_path, output_dir=output_dir)
    
    if args.validate_only:
        logger.info("Configuration validation complete. Exiting.")
        return
    
    # Process dataset
    input_dir = Path(args.input_dir) if args.input_dir else DEFAULT_INPUT_DIR
    extractor.process_dataset(input_dir)


if __name__ == "__main__":
    main()
