#!/usr/bin/env python3
"""
Fast Pancreas X-Level Viewer
Streamlined viewer for quickly setting X-coordinate cuts on pancreas segmentations
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
import os
import base64
import io
from PIL import Image
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import json
import multiprocessing
from datetime import datetime
from scipy import ndimage

app = Flask(__name__)
CORS(app)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (defaults to this repo layout; override via env or config.json)
# Root is the repo directory containing this app
REPO_ROOT = Path(__file__).resolve().parents[1]

def _env_path(var: str, default: Path) -> Path:
    val = os.environ.get(var)
    return Path(val) if val else default

# CT volume(s) are expected in '<repo>/niftii' by default
CT_DIR = _env_path("RADPANC_CT_DIR", REPO_ROOT / "niftii")

# Organ segmentations are expected in '<repo>/data/pancreas'
PANCREAS_SEG_DIR = _env_path("RADPANC_SEG_DIR", REPO_ROOT / "data" / "pancreas")

# Optional: if present, these will be used; otherwise ignored
HEAD_MASKS_DIR = _env_path("RADPANC_HEAD_MASKS_DIR", REPO_ROOT / "data" / "head")
SCANNER_METADATA_PATH = _env_path("RADPANC_SCANNER_META", REPO_ROOT / "scanner_metadata.csv")

COMPLETED_PATIENTS_PATH = Path(__file__).parent / "completed_patients.json"
X_COORDINATE_SELECTIONS_PATH = _env_path(
    "RADPANC_COORDINATES_PATH", Path(__file__).parent / "x_coordinate_selections.json"
)

def find_pancreas_seg(patient_name: str):
    """Resolve a pancreas segmentation path for a given patient.

    Tries common layouts:
    - <seg_dir>/<patient>/pancreas.nii.gz
    - <seg_dir>/<patient>_pancreas.nii.gz
    - <seg_dir>/pancreas.nii.gz
    - Any file in <seg_dir> that contains 'pancreas' in its name
    Returns a Path or None.
    """
    candidates = [
        PANCREAS_SEG_DIR / patient_name / "pancreas.nii.gz",
        PANCREAS_SEG_DIR / f"{patient_name}_pancreas.nii.gz",
        PANCREAS_SEG_DIR / "pancreas.nii.gz",
        CT_DIR / patient_name / "pancreas.nii.gz",
        CT_DIR / f"{patient_name}_pancreas.nii.gz",
    ]
    for p in candidates:
        if p.exists():
            return p
    # If a generic pancreas mask is colocated with CTs (e.g., test case),
    # use it only when it matches the CT voxel grid.
    generic_seg = CT_DIR / "pancreas.nii.gz"
    ct_path = CT_DIR / f"{patient_name}.nii.gz"
    if generic_seg.exists() and ct_path.exists():
        try:
            ct_nii = nib.load(ct_path)
            seg_nii = nib.load(generic_seg)
            if ct_nii.shape == seg_nii.shape and np.allclose(ct_nii.affine, seg_nii.affine):
                return generic_seg
        except Exception:
            pass
    # Fallback: any pancreas-like file in seg dir
    for p in sorted(PANCREAS_SEG_DIR.glob("*pancreas*.nii*")):
        if p.exists():
            return p
    return None

# Global cache for loaded data with multi-core processing
data_cache = {}
slice_cache = {}
try:
    executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
except Exception:
    # Fallback when process pools are restricted (e.g., on constrained systems)
    executor = ThreadPoolExecutor(max_workers=4)

class FastNiftiLoader:
    """Optimized NIfTI loader with lazy loading and caching"""
    
    @staticmethod
    def get_slice_lazy(file_path, slice_idx, axis='axial'):
        """Load a single slice without loading entire volume"""
        # Use nibabel's proxy to avoid loading full data
        nii_proxy = nib.load(file_path)
        
        # Get data shape without loading
        shape = nii_proxy.shape
        
        # Load only the required slice
        if axis == 'axial':
            if slice_idx >= shape[2]:
                slice_idx = shape[2] - 1
            slice_data = nii_proxy.dataobj[:, :, slice_idx]
        elif axis == 'sagittal':
            if slice_idx >= shape[0]:
                slice_idx = shape[0] - 1
            slice_data = nii_proxy.dataobj[slice_idx, :, :]
        else:  # coronal
            if slice_idx >= shape[1]:
                slice_idx = shape[1] - 1
            slice_data = nii_proxy.dataobj[:, slice_idx, :]
            
        arr = np.array(slice_data).T
        # Conventional axial display (anterior up): flip vertical axis.
        if axis == 'axial':
            arr = np.flipud(arr)
        return arr, nii_proxy.affine
    
    @staticmethod
    def get_pancreas_bounds(seg_path):
        """Get slice bounds containing pancreas without loading full volume"""
        nii = nib.load(seg_path)
        data = nii.get_fdata()
        
        # Find slices with pancreas
        z_slices = np.any(data > 0, axis=(0, 1))
        if np.any(z_slices):
            z_indices = np.where(z_slices)[0]
            return {
                'min_slice': int(z_indices[0]),
                'max_slice': int(z_indices[-1]),
                'center_slice': int((z_indices[0] + z_indices[-1]) // 2)
            }
        return {'min_slice': 0, 'max_slice': 0, 'center_slice': 0}

def load_scanner_metadata():
    """Load scanner metadata"""
    if SCANNER_METADATA_PATH.exists():
        return pd.read_csv(SCANNER_METADATA_PATH)
    return pd.DataFrame()

def should_apply_x_cut(current_z, z_limit):
    """
    Determine if X-cut should be applied at current Z level.
    X-cut is applied ABOVE the z_limit.
    """
    if z_limit is None:
        return True  # No limit set, apply everywhere
    return current_z > z_limit


def create_overlay_fast(ct_slice, seg_slice, x_line=None, y_line=None, affine=None, current_z=None, z_limit=None):
    """Create overlay with CT, segmentation, X/Y-lines with Z-limit"""
    # Debug logging
    logger.info(f"create_overlay_fast called: current_z={current_z}, z_limit={z_limit}, x_line={x_line}, y_line={y_line}")
    
    # Window the CT data (soft tissue window)
    ct_windowed = np.clip((ct_slice + 150) / 400, 0, 1)
    
    # Create RGB image
    height, width = ct_slice.shape
    rgb_image = np.zeros((height, width, 3))
    rgb_image[:, :, :] = ct_windowed[:, :, np.newaxis]
    
    # Always start with green pancreas (what will be kept)
    if seg_slice is not None and np.any(seg_slice > 0):
        mask = seg_slice > 0
        rgb_image[:, :, 0][mask] = 0.0  # No red
        rgb_image[:, :, 1][mask] = 1.0  # Full green
        rgb_image[:, :, 2][mask] = 0.0  # No blue (pure green)
    
    # Add X/Y-lines and cut preview if specified
    if (x_line is not None or y_line is not None) and affine is not None:
        # Convert world X coordinate to pixel coordinate using affine matrix
        # For axial view, X corresponds to the first dimension (columns)
        # We need to find which pixel column corresponds to the given X world coordinate
        
        # Create array of pixel coordinates for a vertical line
        pixel_coords = np.zeros((height, 4))
        pixel_coords[:, 1] = np.arange(height)  # y varies
        pixel_coords[:, 2] = 0  # z is fixed for this slice
        pixel_coords[:, 3] = 1  # homogeneous coordinate
        
        # Find the pixel x that gives us the closest world X
        best_pixel_x = None
        if x_line is not None:
            min_diff = float('inf')
            for px in range(width):
                pixel_coords[:, 0] = px
                world_coords = np.dot(affine, pixel_coords.T).T
                avg_world_x = np.mean(world_coords[:, 0])
                diff = abs(avg_world_x - x_line)
                if diff < min_diff:
                    min_diff = diff
                    best_pixel_x = px
        
        # Find the pixel y that gives us the closest world Y
        best_pixel_y = None
        if y_line is not None:
            min_diff = float('inf')
            # Create array for horizontal line
            pixel_coords_h = np.zeros((width, 4))
            pixel_coords_h[:, 0] = np.arange(width)  # x varies
            pixel_coords_h[:, 2] = 0  # z is fixed
            pixel_coords_h[:, 3] = 1
            
            for py in range(height):
                pixel_coords_h[:, 1] = py
                world_coords = np.dot(affine, pixel_coords_h.T).T
                avg_world_y = np.mean(world_coords[:, 1])
                diff = abs(avg_world_y - y_line)
                if diff < min_diff:
                    min_diff = diff
                    best_pixel_y = py
        
        # Show what would be cut based on X, Y, and Z lines
        if seg_slice is not None and (best_pixel_x is not None or best_pixel_y is not None):
            cut_count = 0
            protected_count = 0
            
            for y in range(height):
                for x in range(width):
                    if seg_slice[y, x] > 0:
                        # IMPORTANT: In medical imaging, patient RIGHT is image LEFT
                        # X-line cuts the tail (patient's LEFT side, which is image RIGHT)
                        if best_pixel_x is not None and x < best_pixel_x:
                            # LEFT of X-line in image = RIGHT anatomical side = HEAD
                            # This is ALWAYS KEPT (stays green) - X-line overrides everything
                            pass  # Keep the green color
                        elif best_pixel_x is not None and x >= best_pixel_x:
                            # RIGHT of X-line in image = LEFT anatomical side = TAIL
                            # Need to check Y position and Z-limit
                            
                            if best_pixel_y is not None and y >= best_pixel_y:
                                # POSTERIOR to Y-line (below in image) - check Z-limit
                                if current_z is not None and z_limit is not None and current_z <= z_limit:
                                    # Below/at Z-limit AND posterior to Y - PROTECTED
                                    rgb_image[y, x, 0] = 0.0  # No red
                                    rgb_image[y, x, 1] = 1.0  # Full green
                                    rgb_image[y, x, 2] = 0.3  # Slight blue (bright green)
                                    protected_count += 1
                                else:
                                    # Above Z-limit or no Z-limit - remove
                                    rgb_image[y, x, 0] = 1.0  # Full red
                                    rgb_image[y, x, 1] = 0.2  # Slight green for visibility
                                    rgb_image[y, x, 2] = 0.2  # Slight blue for visibility
                                    cut_count += 1
                            elif best_pixel_y is None:
                                # No Y-line specified - remove all tail
                                rgb_image[y, x, 0] = 1.0  # Full red
                                rgb_image[y, x, 1] = 0.2  # Slight green for visibility
                                rgb_image[y, x, 2] = 0.2  # Slight blue for visibility
                                cut_count += 1
                            else:
                                # ANTERIOR to Y-line (above in image) - always remove (no Z protection)
                                rgb_image[y, x, 0] = 1.0  # Full red
                                rgb_image[y, x, 1] = 0.2  # Slight green for visibility
                                rgb_image[y, x, 2] = 0.2  # Slight blue for visibility
                                cut_count += 1
            
            if cut_count > 0:
                logger.info(f"  {cut_count} pixels marked for removal (bright red)")
            if protected_count > 0:
                logger.info(f"  {protected_count} pixels protected by Z-limit (bright green)")
        
        # Draw the X-line (make it more visible)
        if best_pixel_x is not None and 0 <= best_pixel_x < width:
            # Make line 3 pixels wide for better visibility
            for offset in range(-1, 2):
                x_pos = best_pixel_x + offset
                if 0 <= x_pos < width:
                    # Draw line with alternating colors for visibility
                    if current_z is not None and z_limit is not None and current_z <= z_limit:
                        # Below Z-limit: draw line in cyan to show it's inactive
                        rgb_image[:, x_pos, 0] = 0.0  # No red
                        rgb_image[:, x_pos, 1] = 1.0  # Full green
                        rgb_image[:, x_pos, 2] = 1.0  # Full blue (cyan)
                    else:
                        # Above Z-limit or no limit: draw line in red
                        rgb_image[:, x_pos, 0] = 1.0  # Full red
                        rgb_image[:, x_pos, 1] = 0.0  # No green
                        rgb_image[:, x_pos, 2] = 0.0  # No blue
        
        # Draw the Y-line (make it more visible)
        if best_pixel_y is not None and 0 <= best_pixel_y < height:
            # Make line 3 pixels wide for better visibility
            for offset in range(-1, 2):
                y_pos = best_pixel_y + offset
                if 0 <= y_pos < height:
                    # Draw line with same color scheme as X-line
                    if current_z is not None and z_limit is not None and current_z <= z_limit:
                        # Below Z-limit: draw line in cyan to show it's inactive
                        rgb_image[y_pos, :, 0] = 0.0  # No red
                        rgb_image[y_pos, :, 1] = 1.0  # Full green
                        rgb_image[y_pos, :, 2] = 1.0  # Full blue (cyan)
                    else:
                        # Above Z-limit or no limit: draw line in red (same as X-line)
                        rgb_image[y_pos, :, 0] = 1.0  # Full red
                        rgb_image[y_pos, :, 1] = 0.0  # No green
                        rgb_image[y_pos, :, 2] = 0.0  # No blue
    
    return rgb_image

def array_to_base64_fast(array):
    """Fast conversion to base64"""
    img_array = (array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format='PNG', optimize=False)  # No optimization for speed
    buffer.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"

@app.route('/')
def index():
    """Serve the main interface"""
    return render_template('viewer.html')

def load_completed_patients():
    """Load list of completed patients"""
    if COMPLETED_PATIENTS_PATH.exists():
        with open(COMPLETED_PATIENTS_PATH, 'r') as f:
            return json.load(f)
    return {}

@app.route('/api/patients')
def get_patients_fast():
    """Get all patients with completion status"""
    scanner_metadata = load_scanner_metadata()
    completed_patients = load_completed_patients()
    
    # Load saved measurements to determine completion status
    saved_measurements = {}
    selections_file = X_COORDINATE_SELECTIONS_PATH
    if selections_file.exists():
        with open(selections_file, 'r') as f:
            saved_measurements = json.load(f)
    
    # Get all CT files
    ct_files = sorted(CT_DIR.glob("*.nii.gz"))
    all_patients = []
    
    for ct_file in ct_files:
        patient_name = ct_file.stem.replace('.nii', '')
        
        # Resolve pancreas segmentation path flexibly
        seg_path = find_pancreas_seg(patient_name)
        if not seg_path:
            continue
        
        # Get scanner info
        scanner_info = "Unknown Scanner"
        if not scanner_metadata.empty and patient_name in scanner_metadata['PatientID'].values:
            patient_data = scanner_metadata[scanner_metadata['PatientID'] == patient_name].iloc[0]
            scanner_info = f"{patient_data['Manufacturer']} {patient_data['ManufacturerModelName']}"
        
        # Check if patient has saved data (either in completed_patients.json or x_coordinate_selections.json)
        is_completed = patient_name in completed_patients or patient_name in saved_measurements
        
        all_patients.append({
            'name': patient_name,
            'scanner': scanner_info,
            'completed': is_completed,
            'ct_path': str(ct_file),
            'seg_path': str(seg_path)
        })
    
    return jsonify(all_patients)

@app.route('/api/patient_info/<patient_name>')
def get_patient_info_fast(patient_name):
    """Get basic patient info and pancreas bounds"""
    ct_path = CT_DIR / f"{patient_name}.nii.gz"
    seg_path = find_pancreas_seg(patient_name)
    
    if not ct_path.exists() or not seg_path or not Path(seg_path).exists():
        return jsonify({'error': 'Patient data not found'}), 404
    
    # Get CT dimensions and affine
    ct_proxy = nib.load(ct_path)
    ct_shape = ct_proxy.shape
    affine = ct_proxy.affine
    
    # Get pancreas bounds
    bounds = FastNiftiLoader.get_pancreas_bounds(seg_path)
    
    # Get existing head extraction info if available
    head_info = {}
    head_dir = HEAD_MASKS_DIR / patient_name
    if head_dir.exists():
        info_file = head_dir / "extraction_info.json"
        if info_file.exists():
            with open(info_file, 'r') as f:
                head_info = json.load(f)
    
    # Get saved measurements if available
    saved_measurements = {}
    selections_file = X_COORDINATE_SELECTIONS_PATH
    if selections_file.exists():
        with open(selections_file, 'r') as f:
            all_saved = json.load(f)
            if patient_name in all_saved:
                saved_measurements = all_saved[patient_name]
    
    return jsonify({
        'dimensions': list(ct_shape),
        'affine': affine.tolist(),
        'pancreas_bounds': bounds,
        'head_extraction': head_info,
        'saved_measurements': saved_measurements
    })

@app.route('/api/slice/<patient_name>/<int:slice_idx>')
def get_slice_fast(patient_name, slice_idx):
    """Get a single slice with overlay"""
    ct_path = CT_DIR / f"{patient_name}.nii.gz"
    seg_path = find_pancreas_seg(patient_name)
    
    x_coord = request.args.get('x_coord', type=float)
    y_coord = request.args.get('y_coord', type=float)
    z_limit = request.args.get('z_limit', type=int)
    show_mask = request.args.get('show_mask', 'true').lower() == 'true'
    
    try:
        # Check if data is preloaded
        cache_key = f"{patient_name}_full"
        if cache_key in data_cache:
            # Use cached data for faster access
            cached = data_cache[cache_key]
            ct_slice = np.flipud(cached['ct_data'][:, :, slice_idx].T)
            affine = cached['affine']
            seg_slice = None
            if show_mask:
                seg_slice = np.flipud(cached['seg_data'][:, :, slice_idx].T)
        else:
            # Load slices lazily
            ct_slice, affine = FastNiftiLoader.get_slice_lazy(ct_path, slice_idx)
            
            # Only load segmentation if needed
            seg_slice = None
            if show_mask:
                if not seg_path:
                    return jsonify({'error': 'Segmentation not found'}), 404
                seg_slice, _ = FastNiftiLoader.get_slice_lazy(seg_path, slice_idx)
        
        # Create overlay
        overlay = create_overlay_fast(ct_slice, seg_slice, x_coord, y_coord, affine, slice_idx, z_limit)
        
        # Convert to base64
        img_data = array_to_base64_fast(overlay)
        
        # Calculate world coordinates for center of slice
        i, j, k = ct_slice.shape[0] // 2, ct_slice.shape[1] // 2, slice_idx
        voxel = np.array([i, j, k, 1])
        world = affine @ voxel
        
        # Also return the affine matrix for client-side calculations
        return jsonify({
            'image': img_data,
            'slice_idx': slice_idx,
            'world_coords': {
                'x': float(world[0]),
                'y': float(world[1]),
                'z': float(world[2])
            },
            'affine': affine.tolist(),
            'shape': [int(ct_slice.shape[0]), int(ct_slice.shape[1])]
        })
    except Exception as e:
        logger.error(f"Error loading slice: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/x_analysis/<patient_name>')
def analyze_x_distribution(patient_name):
    """Analyze X-coordinate distribution in pancreas"""
    seg_path = find_pancreas_seg(patient_name)
    
    if not seg_path or not Path(seg_path).exists():
        return jsonify({'error': 'Segmentation not found'}), 404
    
    # Load segmentation
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata()
    affine = seg_nii.affine
    
    # Find all voxels in pancreas
    pancreas_voxels = np.argwhere(seg_data > 0)
    
    # Convert to world coordinates
    x_coords = []
    for voxel in pancreas_voxels:
        world = affine @ np.append(voxel, 1)
        x_coords.append(world[0])
    
    x_coords = np.array(x_coords)
    
    # Calculate statistics
    return jsonify({
        'x_min': float(np.min(x_coords)),
        'x_max': float(np.max(x_coords)),
        'x_mean': float(np.mean(x_coords)),
        'x_median': float(np.median(x_coords)),
        'x_std': float(np.std(x_coords)),
        'histogram': {
            'bins': np.linspace(x_coords.min(), x_coords.max(), 20).tolist(),
            'counts': np.histogram(x_coords, bins=20)[0].tolist()
        }
    })

@app.route('/api/quick_save', methods=['POST'])
def save_x_selections():
    """Save X-coordinate selections for batch processing"""
    data = request.json
    
    output_file = X_COORDINATE_SELECTIONS_PATH
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing data if file exists
    existing_data = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            existing_data = json.load(f)
    
    # Update with new selection including measurements, z_limit, and y_coordinate
    existing_data[data['patient']] = {
        'x_coordinate': data['x_coordinate'],
        'y_coordinate': data.get('y_coordinate', None),
        'z_limit': data.get('z_limit', None),
        'scanner': data.get('scanner', 'Unknown'),
        'timestamp': data.get('timestamp', ''),
        'slice_number': data.get('slice_number', 0),
        'neck_thickness_mm': data.get('neck_thickness', None),
        'duct_size_mm': data.get('duct_size', None)
    }
    
    # Save back
    with open(output_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    
    return jsonify({'status': 'saved', 'total_saved': len(existing_data)})

@app.route('/api/mark_complete/<patient_name>', methods=['POST'])
def mark_patient_complete(patient_name):
    """Mark a patient as completed"""
    completed = load_completed_patients()
    
    if patient_name not in completed:
        completed[patient_name] = {
            'timestamp': datetime.now().isoformat(),
            'completed': True
        }
        
        with open(COMPLETED_PATIENTS_PATH, 'w') as f:
            json.dump(completed, f, indent=2)
    
    return jsonify({'status': 'marked_complete'})


@app.route('/api/z_limit_analysis/<patient_name>')
def analyze_z_limit(patient_name):
    """Analyze impact of Z-limit on pancreatic head extraction."""
    seg_path = find_pancreas_seg(patient_name)
    x_coord = request.args.get('x_coord', type=float)
    z_limit = request.args.get('z_limit', type=int)
    
    if not seg_path or not Path(seg_path).exists() or x_coord is None:
        return jsonify({'error': 'Invalid parameters'}), 400
    
    # Load segmentation
    seg_nii = nib.load(seg_path)
    seg_data = seg_nii.get_fdata()
    affine = seg_nii.affine
    
    # Calculate statistics
    total_pancreas_voxels = np.sum(seg_data > 0)
    preserved_voxels = 0
    cut_voxels = 0
    
    # Analyze each slice
    for z in range(seg_data.shape[2]):
        slice_data = seg_data[:, :, z]
        if np.any(slice_data > 0):
            if z_limit is not None and z <= z_limit:
                # Below Z-limit: preserve all
                preserved_voxels += np.sum(slice_data > 0)
            else:
                # Above Z-limit: apply X-cut
                for voxel in np.argwhere(slice_data > 0):
                    world_coord = affine @ np.append([voxel[0], voxel[1], z], 1)
                    if world_coord[0] < x_coord:  # Right of cut (keep)
                        preserved_voxels += 1
                    else:  # Left of cut (remove)
                        cut_voxels += 1
    
    preservation_rate = (preserved_voxels / total_pancreas_voxels * 100) if total_pancreas_voxels > 0 else 0
    
    return jsonify({
        'total_pancreas_voxels': int(total_pancreas_voxels),
        'preserved_voxels': int(preserved_voxels),
        'cut_voxels': int(cut_voxels),
        'preservation_rate': float(preservation_rate),
        'z_limit': z_limit
    })

@app.route('/api/preload/<patient_name>')
def preload_patient_data(patient_name):
    """Preload all slices for a patient using multi-core processing"""
    ct_path = CT_DIR / f"{patient_name}.nii.gz"
    seg_path = find_pancreas_seg(patient_name)
    
    if not ct_path.exists() or not seg_path or not Path(seg_path).exists():
        return jsonify({'error': 'Patient data not found'}), 404
    
    # Load full volumes into cache
    cache_key = f"{patient_name}_full"
    if cache_key not in data_cache:
        logger.info(f"Preloading full data for {patient_name}")
        ct_nii = nib.load(ct_path)
        seg_nii = nib.load(seg_path)
        
        data_cache[cache_key] = {
            'ct_data': ct_nii.get_fdata(),
            'seg_data': seg_nii.get_fdata(),
            'affine': ct_nii.affine,
            'shape': ct_nii.shape
        }
    
    return jsonify({
        'status': 'preloaded',
        'slices': data_cache[cache_key]['shape'][2]
    })

if __name__ == '__main__':
    print("Starting Fast Pancreas X-Level Viewer...")
    print(f"CT directory: {CT_DIR}")
    print(f"Segmentation directory: {PANCREAS_SEG_DIR}")
    host = os.environ.get("RADPANC_VIEWER_HOST", "127.0.0.1")
    port = int(os.environ.get("RADPANC_VIEWER_PORT", "5003"))
    print(f"Access the viewer at: http://{host}:{port}")
    
    app.run(debug=True, host=host, port=port)
