import numpy as np
from PIL import Image, ImageDraw
import os
import random
import math
import cv2 # Import OpenCV

GAUSSIAN_SIGMA = 3 # Standard deviation for Gaussian blur
POINT_RADIUS = 2 # Radius of the circle drawn for each point

def create_image(points, img_size=(64, 64), background_color=(0, 0, 0)):
    # Create a black PIL Image
    img = Image.new('L', img_size, color=0) # Grayscale image, black background
    draw = ImageDraw.Draw(img)

    # Draw a white circle at each point
    for p in points:
        x, y = int(p[0]), int(p[1])
        # Ensure points are within bounds
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            draw.ellipse((x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS), fill=255) # White circle

    # Convert PIL Image to NumPy array for OpenCV processing
    img_array = np.array(img)

    # Apply Gaussian blur to create heatmap
    # Kernel size should be odd, and typically related to sigma (e.g., 6*sigma + 1)
    ksize = int(6 * GAUSSIAN_SIGMA + 1) # Ensure ksize is odd
    if ksize % 2 == 0: ksize += 1
    blurred_array = cv2.GaussianBlur(img_array, (ksize, ksize), GAUSSIAN_SIGMA)

    # Convert back to PIL Image (grayscale 'L' mode)
    img = Image.fromarray(blurred_array, mode='L')
    return img

def add_noise(points, noise_level=5):
    noisy_points = []
    for x, y in points:
        x_noisy = x + random.uniform(-noise_level, noise_level)
        y_noisy = y + random.uniform(-noise_level, noise_level)
        noisy_points.append((x_noisy, y_noisy))
    return noisy_points

def transform_points(points, translation, angle_deg, scale):
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    
    transformed_points = []
    for x, y in points:
        # Scale
        sx, sy = x * scale, y * scale
        
        # Rotate
        rx = sx * cos_a - sy * sin_a
        ry = sx * sin_a + sy * cos_a
        
        # Translate
        transformed_points.append((rx + translation[0], ry + translation[1]))
        
    return transformed_points

# Helper function to get a point on a Bezier curve
def bezier_point(t, control_points):
    n = len(control_points) - 1
    point = np.zeros(2)
    for i in range(n + 1):
        bernstein_coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        point += bernstein_coeff * np.array(control_points[i])
    return point.tolist()

def generate_v_shape_points(num_points, img_size):
    # Constraint: angle between arms is 20-70 degrees
    angle_deg = random.uniform(20, 70)
    half_angle_rad = np.deg2rad(angle_deg / 2)
    
    # Base lengths for each arm, allowing different lengths
    length_left = img_size[0] * random.uniform(0.15, 0.55) # Expanded range
    length_right = img_size[0] * random.uniform(0.15, 0.55) # Expanded range

    # Apex at origin initially
    apex_x, apex_y = 0, 0
    
    points = [(apex_x, apex_y)]
    
    # Distribute points between the two arms, ensuring at least one point per arm if possible
    num_points_on_arms = num_points - 1 # Exclude apex point
    if num_points_on_arms > 0:
        # Ensure at least one point on each arm if num_points_on_arms >= 2
        if num_points_on_arms >= 2:
            num_points_left_arm = random.randint(1, num_points_on_arms - 1) 
        else: # num_points_on_arms is 1 (num_points is 2), so all points go to one arm
            num_points_left_arm = 1
        num_points_right_arm = num_points_on_arms - num_points_left_arm
    else:
        num_points_left_arm = 0
        num_points_right_arm = 0

    # Left arm
    if num_points_left_arm > 0:
        for i in range(1, num_points_left_arm + 1):
            dist = length_left * (i / num_points_left_arm)
            x = apex_x - dist * np.sin(half_angle_rad)
            y = apex_y - dist * np.cos(half_angle_rad)
            points.append((x, y))

    # Right arm
    if num_points_right_arm > 0:
        for i in range(1, num_points_right_arm + 1):
            dist = length_right * (i / num_points_right_arm)
            x = apex_x + dist * np.sin(half_angle_rad)
            y = apex_y - dist * np.cos(half_angle_rad)
            points.append((x, y))

    # Augmentations
    scale = random.uniform(0.5, 1.5) # Expanded range
    rotation = random.uniform(0, 360)
    translation = (img_size[0] / 2 + random.uniform(-20, 20), img_size[1] / 2 + random.uniform(-20, 20)) # Expanded range

    return transform_points(points, translation, rotation, scale)

def generate_ring_points(num_points, img_size, fill_factor=0.0):
    # Constraint: Ring must not go outside image boundaries
    scale = random.uniform(0.3, 1.1) # Expanded range for outer_radius relative to max possible
    outer_radius = min(img_size) / 2 * scale

    buffer = 2 # Buffer from the edge
    center_x = random.uniform(outer_radius + buffer, img_size[0] - outer_radius - buffer)
    center_y = random.uniform(outer_radius + buffer, img_size[1] - outer_radius - buffer)

    points = []

    # Always generate a base outer ring (e.g., 50% of points for the outer ring)
    num_base_outer_points = max(1, int(num_points * 0.5))
    for i in range(num_base_outer_points):
        angle = 2 * np.pi * i / num_base_outer_points
        x = center_x + outer_radius * np.cos(angle)
        y = center_y + outer_radius * np.sin(angle)
        points.append((x, y))

    # Generate inner points based on fill_factor
    num_remaining_points = num_points - num_base_outer_points
    if num_remaining_points > 0:
        # The inner radius for point generation
        # If fill_factor is 0, inner_gen_radius_min is outer_radius (no inner points)
        # If fill_factor is 1, inner_gen_radius_min is 0 (points from center to outer)
        inner_gen_radius_min = outer_radius * (1.0 - fill_factor)

        for _ in range(num_remaining_points):
            angle = random.uniform(0, 2 * np.pi)
            current_radius = random.uniform(inner_gen_radius_min, outer_radius)
            x = center_x + current_radius * np.cos(angle)
            y = center_y + current_radius * np.sin(angle)
            points.append((x, y))
    
    return points

def generate_linear_points(num_points, img_size):
    curve_type = random.choice(['straight', 'quadratic', 'cubic'])
    points = []

    if curve_type == 'straight':
        # Generate a straight line
        length = img_size[0] * random.uniform(0.3, 1.1) # Expanded range
        x_start, y_start = -length / 2, 0
        x_end, y_end = length / 2, 0
        
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.5
            x = x_start + t * (x_end - x_start)
            y = y_start + t * (y_end - y_start)
            points.append((x,y))

    elif curve_type == 'quadratic':
        # Generate a quadratic Bezier curve (one bend)
        # Control points relative to image center for easier scaling/translation
        cp0 = (-img_size[0] * random.uniform(0.2, 0.4), random.uniform(-img_size[1] * 0.05, img_size[1] * 0.05)) # Start point, reduced y-range
        cp1 = (random.uniform(-img_size[0] * 0.1, img_size[0] * 0.1), random.uniform(-img_size[1] * 0.1, img_size[1] * 0.1)) # Control point, reduced y-range
        cp2 = (img_size[0] * random.uniform(0.2, 0.4), random.uniform(-img_size[1] * 0.05, img_size[1] * 0.05)) # End point, reduced y-range
        control_points = [cp0, cp1, cp2]

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.5
            points.append(bezier_point(t, control_points))

    elif curve_type == 'cubic':
        # Generate a cubic Bezier curve (two bends)
        cp0 = (-img_size[0] * random.uniform(0.2, 0.4), random.uniform(-img_size[1] * 0.05, img_size[1] * 0.05)) # Start point, reduced y-range
        cp1 = (random.uniform(-img_size[0] * 0.3, -img_size[0] * 0.1), random.uniform(-img_size[1] * 0.1, img_size[1] * 0.1)) # Control point 1, reduced y-range
        cp2 = (random.uniform(img_size[0] * 0.1, img_size[0] * 0.3), random.uniform(-img_size[1] * 0.1, img_size[1] * 0.1)) # Control point 2, reduced y-range
        cp3 = (img_size[0] * random.uniform(0.2, 0.4), random.uniform(-img_size[1] * 0.05, img_size[1] * 0.05)) # End point, reduced y-range
        control_points = [cp0, cp1, cp2, cp3]

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0.5
            points.append(bezier_point(t, control_points))

    # Augmentations (applied to all curve types)
    scale = random.uniform(0.6, 1.4) # Expanded range
    rotation = random.uniform(0, 360)
    translation = (img_size[0] / 2 + random.uniform(-25, 25), img_size[1] / 2 + random.uniform(-25, 25)) # Expanded range
    
    return transform_points(points, translation, rotation, scale)

def generate_dispersed_points(num_points, img_size):
    points = []
    for _ in range(num_points):
        x = random.uniform(0, img_size[0])
        y = random.uniform(0, img_size[1])
        points.append((x, y))
    return points

def main():
    output_dir = "train_flock_analyzer/dataset"
    classes = {"V-shape": generate_v_shape_points, 
               "Line": generate_linear_points, 
               "Ring": generate_ring_points, 
               "dispersed": generate_dispersed_points}
    
    num_images_per_class = 1000
    img_size = (64, 64)
    min_points = 5
    max_points = 20

    for class_name, generator_func in classes.items():
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        print(f"Generating {num_images_per_class} images for class: {class_name}")

        for i in range(num_images_per_class):
            num_points = random.randint(min_points, max_points)
            
            # Generate points using the assigned function
            if class_name == "Ring":
                fill_factor = random.uniform(0.0, 1.0) # Random fill factor for each Ring image
                points = generator_func(num_points, img_size, fill_factor=fill_factor)
            else:
                points = generator_func(num_points, img_size)
            
            # Add noise to all point sets
            noisy_points = add_noise(points, noise_level=1)

            if noisy_points:
                img = create_image(noisy_points, img_size=img_size)
                img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.png"))

if __name__ == "__main__":
    main()