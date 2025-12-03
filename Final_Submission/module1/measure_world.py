import numpy as np

camera_matrix = np.array([
    [961.461004058896, 0, 618.9908302325701],
    [0, 964.8013393433856, 342.0805337054228],
    [0, 0, 1]
])

calib_width, calib_height = 1280, 720   # original calibration resolution
test_width, test_height = 2220, 1480    # rubik's cube image resolution

# rescale matrix 
scale_x = test_width / calib_width
scale_y = test_height / calib_height
camera_matrix[0, 0] *= scale_x  # fx
camera_matrix[0, 2] *= scale_x  # cx
camera_matrix[1, 1] *= scale_y  # fy
camera_matrix[1, 2] *= scale_y  # cy

print("Updated camera matrix:\n", camera_matrix)

#clicked points from last lab/class
x1, y1 = 974, 958
x2, y2 = 1241, 959

#z value  in cm
Z = 34.0

# extract intrinsics
fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]

dx_pix = abs(x2 - x1)
dy_pix = abs(y2 - y1)

# Perspective projection equations
dx_world = (dx_pix * Z) / fx
dy_world = (dy_pix * Z) / fy
diag = np.sqrt(dx_world**2 + dy_world**2)

print("Camera Dimension Measurement:")
print(f"Clicked Points: P1=({x1},{y1}), P2=({x2},{y2})")
print(f"Z = {Z:.2f} cm")
print(f"Δx_pixels = {dx_pix}, Δy_pixels = {dy_pix}")
print(" ")
print(f"ΔX_world = {dx_world:.4f} cm")
print(f"ΔY_world = {dy_world:.4f} cm")
print(f"Diagonal distance = {diag:.4f} cm")
print(" ")