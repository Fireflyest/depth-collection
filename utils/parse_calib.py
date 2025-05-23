import numpy as np

def parse_scott_reef_calibration(calib_path):
    """
    Parse camera calibration file for Scott Reef project
    
    Parameters:
        calib_path: Path to calibration file
        
    Returns:
        dict: Dictionary containing parsed parameters
    """
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # First line is the number of cameras
    num_cameras = int(lines[0].strip())
    print(f"Number of cameras: {num_cameras}")
    
    cameras = []
    for i in range(num_cameras):
        if i+2 >= len(lines):
            break
            
        params = lines[i+2].strip().split()
        if len(params) < 27:  # Make sure there are enough parameters
            print(f"Warning: Camera {i+1} has insufficient parameters")
            continue
            
        width = int(params[0])
        height = int(params[1])
        
        # Intrinsic matrix (3x3)
        # [fx  0  cx]
        # [0  fy  cy]
        # [0   0   1]
        intrinsic = np.array([
            [float(params[2]), float(params[3]), float(params[4])],
            [float(params[5]), float(params[6]), float(params[7])],
            [float(params[8]), float(params[9]), float(params[10])]
        ])
        
        # Distortion coefficients [k1, k2, p1, p2]
        distortion = np.array([
            float(params[11]), float(params[12]), 
            float(params[13]), float(params[14])
        ])
        
        # Extrinsic matrix (3x4) [R|t]
        extrinsic = np.array([
            [float(params[15]), float(params[16]), float(params[17]), float(params[24])],
            [float(params[18]), float(params[19]), float(params[20]), float(params[25])],
            [float(params[21]), float(params[22]), float(params[23]), float(params[26])]
        ])
        
        # Extract rotation matrix and translation vector
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        
        camera_info = {
            "id": i+1,
            "width": width,
            "height": height,
            "intrinsic": intrinsic,
            "distortion": distortion,
            "extrinsic": extrinsic,
            "rotation": rotation,
            "translation": translation,
            "fx": intrinsic[0, 0],
            "fy": intrinsic[1, 1],
            "cx": intrinsic[0, 2],
            "cy": intrinsic[1, 2],
            "k1": distortion[0],
            "k2": distortion[1],
            "p1": distortion[2],
            "p2": distortion[3]
        }
        cameras.append(camera_info)
    
    # Calculate baseline (stereo baseline) - X-axis translation of the second camera relative to the first
    baseline = None
    if len(cameras) >= 2:
        # Baseline is typically the negative of the second camera's X-axis translation
        baseline = abs(cameras[1]["translation"][0])
    
    return {
        "num_cameras": num_cameras,
        "cameras": cameras,
        "baseline": baseline
    }

if __name__ == "__main__":
    calib_path = "./assets/ScottReef/calib/ScottReef_20090804_084719.calib"
    calib_data = parse_scott_reef_calibration(calib_path)
    
    print("\n== Calibration File Parsing Results ==")
    print(f"Baseline distance: {calib_data['baseline']:.6f} meters")
    
    for i, cam in enumerate(calib_data['cameras']):
        print(f"\n== Camera {i+1} ==")
        print(f"Image size: {cam['width']} x {cam['height']}")
        print(f"Focal length: fx={cam['fx']:.2f}, fy={cam['fy']:.2f}")
        print(f"Principal point: cx={cam['cx']:.2f}, cy={cam['cy']:.2f}")
        print(f"Radial distortion: k1={cam['k1']:.6f}, k2={cam['k2']:.6f}")
        print(f"Tangential distortion: p1={cam['p1']:.6f}, p2={cam['p2']:.6f}")
        
        if i == 0:
            print("Extrinsics: First camera is the reference coordinate system")
        else:
            print(f"Rotation matrix relative to Camera 1:\n{cam['rotation']}")
            print(f"Translation vector relative to Camera 1: [{cam['translation'][0]:.6f}, {cam['translation'][1]:.6f}, {cam['translation'][2]:.6f}]")
