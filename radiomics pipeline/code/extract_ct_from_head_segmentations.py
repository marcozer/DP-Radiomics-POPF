#!/usr/bin/env python3
"""
Extract CT scanner data directly from pancreatic head segmentations
No dilation - just the actual CT values within the segmented head
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (defaults; override via CLI)
BASE_DIR = Path(__file__).resolve().parents[1]
CT_DIR = BASE_DIR / "niftii"
HEAD_DIR = BASE_DIR / "outputs" / "pancreatic_heads_manual_extracted"
OUTPUT_DIR = BASE_DIR / "outputs" / "ct_head_data"

def extract_ct_head_data(patient_name):
    """Extract CT data within pancreatic head segmentation"""
    try:
        # Paths
        ct_path = CT_DIR / f"{patient_name}.nii.gz"
        head_path = HEAD_DIR / patient_name / "pancreatic_head.nii.gz"
        
        if not ct_path.exists():
            logger.warning(f"CT not found for {patient_name}: {ct_path}")
            return None
            
        if not head_path.exists():
            logger.warning(f"Head segmentation not found for {patient_name}: {head_path}")
            return None
        
        # Load data
        ct_nii = nib.load(ct_path)
        head_nii = nib.load(head_path)
        
        ct_data = ct_nii.get_fdata()
        head_mask = head_nii.get_fdata().astype(bool)
        
        # Verify they have the same shape
        if ct_data.shape != head_mask.shape:
            logger.error(f"Shape mismatch for {patient_name}: CT {ct_data.shape} vs Head {head_mask.shape}")
            return None
        
        # Find bounding box of head mask to create cropped volume
        indices = np.argwhere(head_mask)
        if len(indices) == 0:
            logger.warning(f"Empty head mask for {patient_name}")
            return None
        
        # Get bounding box with small padding (3 voxels)
        padding = 3
        min_coords = np.maximum(indices.min(axis=0) - padding, 0)
        max_coords = np.minimum(indices.max(axis=0) + padding + 1, ct_data.shape)
        
        # Create slices for cropping
        crop_slices = tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
        
        # Extract cropped regions
        ct_cropped = ct_data[crop_slices].copy()
        head_cropped = head_mask[crop_slices]
        
        # Apply mask to CT data (set outside head to -1024 HU, which is air)
        ct_head = ct_cropped.copy()
        ct_head[~head_cropped] = -1024
        
        # Calculate statistics for the head region
        head_voxels = head_cropped.sum()
        ct_values_in_head = ct_cropped[head_cropped]
        
        # Update affine matrix for the cropped volume
        new_affine = ct_nii.affine.copy()
        new_affine[:3, 3] = ct_nii.affine[:3, :3] @ min_coords + ct_nii.affine[:3, 3]
        
        # Create output directory
        output_patient_dir = OUTPUT_DIR / patient_name
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cropped CT with head mask applied
        ct_head_nii = nib.Nifti1Image(ct_head, new_affine, ct_nii.header)
        nib.save(ct_head_nii, output_patient_dir / "ct_head.nii.gz")
        
        # Save cropped head mask for radiomics software
        head_cropped_nii = nib.Nifti1Image(head_cropped.astype(np.uint8), new_affine, head_nii.header)
        nib.save(head_cropped_nii, output_patient_dir / "head_mask_cropped.nii.gz")
        
        # Calculate volume
        voxel_volume = np.abs(np.prod(np.diag(ct_nii.affine)[:3]))  # mm³
        head_volume_ml = head_voxels * voxel_volume / 1000
        
        # Save metadata
        metadata = {
            'patient_name': patient_name,
            'original_shape': [int(x) for x in ct_data.shape],
            'cropped_shape': [int(x) for x in ct_cropped.shape],
            'bounding_box': {
                'min': [int(x) for x in min_coords.tolist()],
                'max': [int(x) for x in max_coords.tolist()]
            },
            'head_voxels': int(head_voxels),
            'head_volume_ml': float(head_volume_ml),
            'ct_intensity_stats': {
                'mean': float(np.mean(ct_values_in_head)),
                'std': float(np.std(ct_values_in_head)),
                'min': float(np.min(ct_values_in_head)),
                'max': float(np.max(ct_values_in_head)),
                'median': float(np.median(ct_values_in_head)),
                'percentile_5': float(np.percentile(ct_values_in_head, 5)),
                'percentile_95': float(np.percentile(ct_values_in_head, 95))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Load extraction info if available
        extraction_info_path = HEAD_DIR / patient_name / "extraction_info.json"
        if extraction_info_path.exists():
            with open(extraction_info_path, 'r') as f:
                extraction_info = json.load(f)
                # Convert any numpy types to native Python types
                for key, value in extraction_info.items():
                    if hasattr(value, 'item'):  # numpy scalar
                        extraction_info[key] = value.item()
                metadata['head_extraction_info'] = extraction_info
        
        with open(output_patient_dir / "ct_head_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Processed {patient_name}: {head_voxels} voxels, {head_volume_ml:.1f} mL")
        
        return {
            'patient': patient_name,
            'status': 'success',
            'original_shape': [int(x) for x in ct_data.shape],
            'cropped_shape': [int(x) for x in ct_cropped.shape],
            'head_voxels': int(head_voxels),
            'head_volume_ml': float(head_volume_ml),
            'mean_hu': float(np.mean(ct_values_in_head))
        }
        
    except Exception as e:
        logger.error(f"Error processing {patient_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'patient': patient_name,
            'status': 'error',
            'error': str(e)
        }

def main():
    """Main pipeline to extract CT data from head segmentations"""
    global CT_DIR, HEAD_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description="Extract CT head subvolumes from full CT and head masks")
    parser.add_argument("--ct-dir", type=Path, default=CT_DIR, help="Directory containing CT .nii.gz files")
    parser.add_argument("--head-dir", type=Path, default=HEAD_DIR, help="Directory with head masks per patient")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to write cropped outputs")
    args = parser.parse_args()

    CT_DIR = args.ct_dir
    HEAD_DIR = args.head_dir
    OUTPUT_DIR = args.output_dir

    logger.info("Starting CT extraction from pancreatic head segmentations")
    logger.info("No dilation - extracting exact head regions only")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all patients with head segmentations
    if not HEAD_DIR.exists():
        logger.error(f"Head segmentation directory not found: {HEAD_DIR}")
        return
    
    patient_dirs = [d for d in HEAD_DIR.iterdir() if d.is_dir() and d.name != 'visualizations']
    logger.info(f"Found {len(patient_dirs)} patients with head segmentations")
    
    # Process all patients
    results = []
    for patient_dir in tqdm(patient_dirs, desc="Extracting CT head data"):
        result = extract_ct_head_data(patient_dir.name)
        if result:
            results.append(result)
    
    # Summary
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    logger.info(f"\nCT head extraction complete:")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    
    if successful:
        avg_volume = np.mean([r['head_volume_ml'] for r in successful])
        avg_hu = np.mean([r['mean_hu'] for r in successful])
        logger.info(f"  - Average head volume: {avg_volume:.1f} mL")
        logger.info(f"  - Average CT intensity: {avg_hu:.1f} HU")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_patients': len(patient_dirs),
        'successful': len(successful),
        'failed': len(failed),
        'average_head_volume_ml': float(avg_volume) if successful else 0,
        'average_ct_intensity_hu': float(avg_hu) if successful else 0,
        'results': results
    }
    
    with open(OUTPUT_DIR / 'ct_head_extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if failed:
        logger.warning(f"\nFailed extractions:")
        for r in failed:
            logger.warning(f"  - {r['patient']}: {r.get('error', 'Unknown error')}")
    
    logger.info(f"\nCT head data saved to: {OUTPUT_DIR}")
    logger.info("\nFiles created for each patient:")
    logger.info("  - ct_head.nii.gz: CT intensities within pancreatic head (cropped)")
    logger.info("  - head_mask_cropped.nii.gz: Binary mask for radiomics software")
    logger.info("  - ct_head_metadata.json: Statistics and metadata")
    logger.info("\nThese files are ready for radiomics feature extraction!")

if __name__ == "__main__":
    main()
