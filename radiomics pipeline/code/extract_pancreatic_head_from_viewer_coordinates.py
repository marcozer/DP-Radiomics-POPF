#!/usr/bin/env python3
"""
Extract pancreatic head using coordinates from fast_pancreas_viewer
Uses X, Y, Z coordinates saved during manual segmentation review
"""

import numpy as np
import nibabel as nib
import json
from pathlib import Path
import logging
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration (override via CLI)
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_DIR = REPO_ROOT
CT_DIR = DEFAULT_BASE_DIR / "niftii"
PANCREAS_SEG_DIR = DEFAULT_BASE_DIR / "data" / "pancreas"
COORDINATES_FILE = REPO_ROOT / "pancreas_head_delimiter" / "x_coordinate_selections.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "pancreatic_heads_manual_extracted"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

def _init_worker(ct_dir: Path, seg_dir: Path, coordinates_file: Path, output_dir: Path):
    """Initializer to propagate CLI paths to spawned worker processes (macOS/Windows)."""
    global CT_DIR, PANCREAS_SEG_DIR, COORDINATES_FILE, OUTPUT_DIR, VISUALIZATION_DIR
    CT_DIR = Path(ct_dir)
    PANCREAS_SEG_DIR = Path(seg_dir)
    COORDINATES_FILE = Path(coordinates_file)
    OUTPUT_DIR = Path(output_dir)
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

def load_viewer_coordinates():
    """Load saved coordinates from fast_pancreas_viewer"""
    if not COORDINATES_FILE.exists():
        raise FileNotFoundError(f"Coordinates file not found: {COORDINATES_FILE}")
    
    with open(COORDINATES_FILE, 'r') as f:
        return json.load(f)

def apply_head_extraction_logic(seg_data, affine, x_coord, y_coord=None, z_limit=None):
    """
    Apply the extraction logic from fast_pancreas_viewer
    
    IMPORTANT: Must match the viewer's pixel-based logic exactly!
    The viewer keeps pixels where x < best_pixel_x as HEAD (green)
    """
    head_mask = np.zeros_like(seg_data, dtype=bool)
    
    # Get shape
    nx, ny, nz = seg_data.shape
    
    # For each Z slice, find the best pixel X that matches the world X coordinate
    for k in range(nz):
        # Find any pancreas in this slice
        slice_mask = seg_data[:, :, k] > 0
        if not np.any(slice_mask):
            continue
            
        # Find the pixel x that best matches the world x_coord
        best_pixel_x = None
        min_diff = float('inf')
        
        # Test each pixel column
        for px in range(nx):
            # Get world coordinate for this pixel column at middle of slice
            voxel = np.array([px, ny//2, k, 1])
            world = affine @ voxel
            diff = abs(world[0] - x_coord)
            if diff < min_diff:
                min_diff = diff
                best_pixel_x = px
        
        # Find the pixel y that best matches the world y_coord (if provided)
        best_pixel_y = None
        if y_coord is not None:
            min_diff = float('inf')
            for py in range(ny):
                voxel = np.array([nx//2, py, k, 1])
                world = affine @ voxel
                diff = abs(world[1] - y_coord)
                if diff < min_diff:
                    min_diff = diff
                    best_pixel_y = py
        
        # Apply the cutting logic matching the viewer exactly
        for i in range(nx):
            for j in range(ny):
                if seg_data[i, j, k] > 0:  # Is pancreas
                    # Match viewer logic: x < best_pixel_x is HEAD (kept)
                    if best_pixel_x is not None and i < best_pixel_x:
                        # LEFT of X-line in image = HEAD = ALWAYS KEEP
                        head_mask[i, j, k] = True
                    elif best_pixel_x is not None and i >= best_pixel_x:
                        # RIGHT of X-line in image = TAIL
                        keep_voxel = False
                        
                        if best_pixel_y is not None and j >= best_pixel_y:
                            # POSTERIOR to Y-line (below in image) - check Z-limit
                            if z_limit is not None and k <= z_limit:
                                # Below/at Z-limit AND posterior to Y - KEEP
                                keep_voxel = True
                        
                        head_mask[i, j, k] = keep_voxel
    
    return head_mask

def extract_single_patient(patient_data):
    """Extract pancreatic head for a single patient"""
    patient_name, coords = patient_data
    
    try:
        # Paths
        ct_path = CT_DIR / f"{patient_name}.nii.gz"
        # Resolve pancreas segmentation with flexible layouts
        seg_candidates = [
            PANCREAS_SEG_DIR / patient_name / "pancreas.nii.gz",
            PANCREAS_SEG_DIR / f"{patient_name}_pancreas.nii.gz",
            PANCREAS_SEG_DIR / f"{patient_name}.nii.gz",
        ]
        ct_resolved = ct_path.resolve() if ct_path.exists() else ct_path
        seg_path = None
        for candidate in seg_candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.resolve() == ct_resolved:
                    # Avoid accidentally treating the CT as the segmentation when ct/seg dirs overlap.
                    continue
            except Exception:
                pass
            seg_path = candidate
            break
        if seg_path is None:
            # Generic colocated pancreas mask (e.g., test case)
            generic = PANCREAS_SEG_DIR / "pancreas.nii.gz"
            if generic.exists() and ct_path.exists():
                try:
                    ct_nii = nib.load(ct_path)
                    seg_nii = nib.load(generic)
                    if ct_nii.shape == seg_nii.shape and np.allclose(ct_nii.affine, seg_nii.affine):
                        seg_path = generic
                except Exception:
                    seg_path = None
        
        # Check if files exist
        if not ct_path.exists():
            logger.warning(f"CT not found for {patient_name}: {ct_path}")
            return None
        if seg_path is None or not seg_path.exists():
            logger.warning(f"Segmentation not found for {patient_name}: {seg_path}")
            return None
        
        # Load data
        ct_nii = nib.load(ct_path)
        seg_nii = nib.load(seg_path)
        
        ct_data = ct_nii.get_fdata()
        seg_data = seg_nii.get_fdata()
        affine = seg_nii.affine
        
        # Extract coordinates
        x_coord = coords['x_coordinate']
        y_coord = coords.get('y_coordinate', None)
        z_limit = coords.get('z_limit', None)
        
        y_str = f"{y_coord:.1f}" if y_coord is not None else "None"
        z_str = str(z_limit) if z_limit is not None else "None"
        logger.info(f"Processing {patient_name}: X={x_coord:.1f}, Y={y_str}, Z-limit={z_str}")
        
        # Apply extraction logic
        head_mask = apply_head_extraction_logic(seg_data, affine, x_coord, y_coord, z_limit)
        
        # Calculate statistics
        original_voxels = np.sum(seg_data > 0)
        head_voxels = np.sum(head_mask)
        retention_rate = (head_voxels / original_voxels * 100) if original_voxels > 0 else 0
        
        # Create output directory
        output_patient_dir = OUTPUT_DIR / patient_name
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        
        # Save extracted head
        head_data = seg_data.copy()
        head_data[~head_mask] = 0
        
        head_nii = nib.Nifti1Image(head_data, affine, seg_nii.header)
        nib.save(head_nii, output_patient_dir / "pancreatic_head.nii.gz")
        
        # Save original pancreas for comparison
        nib.save(seg_nii, output_patient_dir / "pancreas.nii.gz")
        
        # Save extraction info
        extraction_info = {
            'patient_name': patient_name,
            'x_coordinate': x_coord,
            'y_coordinate': y_coord,
            'z_limit': z_limit,
            'extraction_method': 'manual_viewer_coordinates',
            'original_voxels': int(original_voxels),
            'head_voxels': int(head_voxels),
            'retention_rate': float(retention_rate),
            'scanner': coords.get('scanner', 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'viewer_timestamp': coords.get('timestamp', 'Unknown'),
            'slice_number': coords.get('slice_number', None)
        }
        
        with open(output_patient_dir / "extraction_info.json", 'w') as f:
            json.dump(extraction_info, f, indent=2)
        
        # Create visualization
        create_visualization(patient_name, ct_data, seg_data, head_mask, affine, 
                           x_coord, y_coord, z_limit, extraction_info)
        
        return {
            'patient': patient_name,
            'status': 'success',
            'retention_rate': float(retention_rate),
            'head_voxels': int(head_voxels)
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

def create_visualization(patient_name, ct_data, seg_data, head_mask, affine, 
                        x_coord, y_coord, z_limit, extraction_info):
    """Create visualization showing the extraction result"""
    try:
        # Find center slice with pancreas
        z_slices = np.any(seg_data > 0, axis=(0, 1))
        if not np.any(z_slices):
            return
        
        z_indices = np.where(z_slices)[0]
        center_z = (z_indices[0] + z_indices[-1]) // 2
        
        # Create figure with multiple views
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Axial views at different levels
        for i, (z_offset, ax) in enumerate([(-10, axes[0, 0]), (0, axes[0, 1]), (10, axes[0, 2])]):
            z = center_z + z_offset
            if 0 <= z < ct_data.shape[2]:
                # CT with overlay
                ct_slice = ct_data[:, :, z].T
                seg_slice = seg_data[:, :, z].T
                head_slice = head_mask[:, :, z].T
                
                # Window CT
                ct_windowed = np.clip((ct_slice + 150) / 400, 0, 1)
                
                # Create overlay
                overlay = np.zeros((*ct_slice.shape, 3))
                overlay[:, :] = ct_windowed[:, :, np.newaxis]
                
                # Original pancreas in yellow
                pancreas_mask = seg_slice > 0
                overlay[pancreas_mask, 0] = 1.0  # Red
                overlay[pancreas_mask, 1] = 1.0  # Green
                overlay[pancreas_mask, 2] = 0.0  # Blue
                
                # Extracted head in green
                head_mask_slice = head_slice > 0
                overlay[head_mask_slice, 0] = 0.0
                overlay[head_mask_slice, 1] = 1.0
                overlay[head_mask_slice, 2] = 0.0
                
                # Show lines if at appropriate Z
                if z_limit is None or z > z_limit:
                    # Find pixel coordinates for lines
                    # X-line
                    best_x_pixel = None
                    min_diff = float('inf')
                    for px in range(ct_slice.shape[1]):
                        voxel = np.array([px, ct_slice.shape[0]//2, z, 1])
                        world = affine @ voxel
                        if abs(world[0] - x_coord) < min_diff:
                            min_diff = abs(world[0] - x_coord)
                            best_x_pixel = px
                    
                    if best_x_pixel is not None:
                        overlay[:, best_x_pixel:best_x_pixel+2, 0] = 1.0
                        overlay[:, best_x_pixel:best_x_pixel+2, 1] = 0.0
                        overlay[:, best_x_pixel:best_x_pixel+2, 2] = 0.0
                
                ax.imshow(overlay)
                ax.set_title(f'Axial Z={z} {"(at Z-limit)" if z == z_limit else ""}')
                ax.axis('off')
        
        # 3D projections
        # Sagittal MIP
        seg_mip_sag = np.max(seg_data, axis=0).T
        head_mip_sag = np.max(head_mask, axis=0).T
        
        axes[1, 0].imshow(seg_mip_sag, cmap='gray', alpha=0.5)
        axes[1, 0].imshow(head_mip_sag, cmap='Greens', alpha=0.7)
        axes[1, 0].set_title('Sagittal MIP (Green=Head)')
        axes[1, 0].axis('off')
        
        # Coronal MIP
        seg_mip_cor = np.max(seg_data, axis=1).T
        head_mip_cor = np.max(head_mask, axis=1).T
        
        axes[1, 1].imshow(seg_mip_cor, cmap='gray', alpha=0.5)
        axes[1, 1].imshow(head_mip_cor, cmap='Greens', alpha=0.7)
        axes[1, 1].set_title('Coronal MIP (Green=Head)')
        axes[1, 1].axis('off')
        
        # Statistics text
        y_str = f"{y_coord:.1f} mm" if y_coord is not None else "None"
        z_str = str(z_limit) if z_limit is not None else "None"
        
        stats_text = f"""Patient: {patient_name}
X-cut: {x_coord:.1f} mm
Y-cut: {y_str}
Z-limit: {z_str}
Scanner: {extraction_info['scanner']}

Original volume: {extraction_info['original_voxels']} voxels
Head volume: {extraction_info['head_voxels']} voxels
Retention: {extraction_info['retention_rate']:.1f}%"""
        
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].axis('off')
        
        plt.suptitle(f'Pancreatic Head Extraction - {patient_name}', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        viz_dir = VISUALIZATION_DIR / patient_name
        viz_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(viz_dir / 'extraction_result.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating visualization for {patient_name}: {str(e)}")

def main():
    """Main extraction pipeline"""
    global CT_DIR, PANCREAS_SEG_DIR, COORDINATES_FILE, OUTPUT_DIR, VISUALIZATION_DIR
    parser = argparse.ArgumentParser(description="Extract pancreatic head subvolumes using viewer coordinates")
    parser.add_argument("--ct-dir", type=Path, default=CT_DIR,
                        help="Directory containing CT NIfTI volumes")
    parser.add_argument("--seg-dir", type=Path, default=PANCREAS_SEG_DIR,
                        help="Directory containing pancreas segmentations")
    parser.add_argument("--coordinates-file", type=Path, default=COORDINATES_FILE,
                        help="JSON with viewer-selected coordinates")
    parser.add_argument(
        "--patient-id",
        action="append",
        default=None,
        help="Only process this patient id (repeatable). Defaults to all patients in the coordinates file.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Directory for cropped head outputs")
    args = parser.parse_args()
    CT_DIR = Path(args.ct_dir)
    PANCREAS_SEG_DIR = Path(args.seg_dir)
    COORDINATES_FILE = Path(args.coordinates_file)
    OUTPUT_DIR = Path(args.output_dir)
    VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

    logger.info("Starting pancreatic head extraction from viewer coordinates")
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load coordinates
    coordinates = load_viewer_coordinates()
    logger.info(f"Loaded coordinates for {len(coordinates)} patients")

    if args.patient_id:
        requested = list(dict.fromkeys(args.patient_id))  # preserve order, de-dup
        missing = [pid for pid in requested if pid not in coordinates]
        if missing:
            logger.error(
                "Requested patient id(s) not found in coordinates file: %s",
                ", ".join(missing),
            )
            return
        coordinates = {pid: coordinates[pid] for pid in requested}
        logger.info("Filtering to %d requested patient(s)", len(coordinates))

    # Prepare patient data
    patient_data = [(patient, coords) for patient, coords in coordinates.items()]
    if not patient_data:
        logger.error("No patients to process after filtering.")
        return
    
    # Process patients in parallel (fallback to threads if process pools are restricted)
    results = []
    max_workers = max(1, min(mp.cpu_count(), len(patient_data)))
    try:
        executor = ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_init_worker,
            initargs=(CT_DIR, PANCREAS_SEG_DIR, COORDINATES_FILE, OUTPUT_DIR),
        )
    except Exception as e:
        logger.warning(f"ProcessPoolExecutor unavailable ({e}); falling back to ThreadPoolExecutor.")
        executor = ThreadPoolExecutor(max_workers=min(4, len(patient_data) or 1))

    with executor:
        with tqdm(total=len(patient_data), desc="Extracting heads") as pbar:
            for result in executor.map(extract_single_patient, patient_data):
                if result:
                    results.append(result)
                pbar.update(1)
    
    # Summary statistics
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    
    logger.info(f"\nExtraction complete:")
    logger.info(f"  - Successful: {len(successful)}")
    logger.info(f"  - Failed: {len(failed)}")
    
    avg_retention = 0.0
    if successful:
        avg_retention = np.mean([r['retention_rate'] for r in successful])
        logger.info(f"  - Average retention rate: {avg_retention:.1f}%")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_patients': len(coordinates),
        'successful': len(successful),
        'failed': len(failed),
        'average_retention_rate': float(avg_retention) if successful else 0,
        'results': results
    }
    
    with open(OUTPUT_DIR / 'extraction_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    if failed:
        logger.warning(f"\nFailed extractions:")
        for r in failed:
            logger.warning(f"  - {r['patient']}: {r.get('error', 'Unknown error')}")
    
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
