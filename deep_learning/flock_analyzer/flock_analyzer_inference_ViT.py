import cv2
import torch
import torch.nn as nn
import time
import numpy as np
from sklearn.cluster import DBSCAN
import os
import argparse
from PIL import Image, ImageDraw
from torchvision import transforms
from torch.utils.data import Dataset # For temporary dataset loading
from scipy.spatial.distance import pdist, squareform # For Line critical points
from scipy.stats import gaussian_kde # For Dispersed critical points
from skimage.morphology import skeletonize # For V-shape critical points

GAUSSIAN_SIGMA = 3 # Standard deviation for Gaussian blur
POINT_RADIUS = 2 # Radius of the circle drawn for each point

# Helper function to find intersection of two lines (defined by two points each)
def _find_line_intersection(line1, line2):
    # Line 1: (x1, y1) to (x2, y2)
    # Line 2: (x3, y3) to (x4, y4)
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if den == 0:
        return None # Lines are parallel or collinear

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    # u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den # Not needed for intersection point

    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)
    return (px, py)

# ROS 메시지 대신 간단한 데이터 구조 사용
class DetectedObject:
    def __init__(self, class_id, prob, x, y, width, height, species_name):
        self.class_id = class_id
        self.prob = prob
        self.box = {'x': x, 'y': y, 'width': width, 'height': height}
        self.species = species_name

class FlockAnalysisResult:
    def __init__(self):
        self.is_flock = False
        self.species = []
        self.number_of_birds = 0
        self.formation = "unknown"
        self.density = 0.0
        self.number_of_clusters = 0
        self.direction = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.image_width = 0
        self.image_height = 0
        self.objects = []
        self.critical_points = [] # Added critical_points attribute
        self.inference_time = 0.0 # Added inference_time attribute

# --- Vision Transformer (ViT) Components (Copied from train_flock_analyzer.py) ---

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x) # (B, E, H', W')
        x = x.flatten(2) # (B, E, N_patches)
        x = x.transpose(1, 2) # (B, N_patches, E)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class FormationClassifier(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=1, num_classes=10, 
                 embed_dim=384, depth=6, num_heads=12, mlp_ratio=4., 
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., 
                 norm_layer=nn.LayerNorm, class_names=None):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.class_names = class_names if class_names is not None else []

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, 
                                          in_channels=in_channels, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize positional embedding and CLS token
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)  # stole cls_token impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def decode_prediction(self, pred):
        predicted_index = torch.argmax(pred, 1).item()
        if predicted_index < len(self.class_names):
            return self.class_names[predicted_index]
        else:
            return f"Unknown class index: {predicted_index}"

# Temporary FormationDataset to load class names
class FormationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = []

        sorted_class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for i, class_name in enumerate(sorted_class_names):
            class_path = os.path.join(root_dir, class_name)
            self.class_to_idx[class_name] = i
            self.idx_to_class.append(class_name)
            # No need to load image paths for this purpose, just class names

    def __len__(self):
        return len(self.image_paths) # Will be 0, but not used for class names

    def __getitem__(self, idx):
        raise NotImplementedError("This dataset is for class name loading only.")


class PseudoFlockAnalyzer:
    def __init__(self, yolo_model_path=None, cnn_model_path=None, dataset_root_dir=None):
        self.yolo_model = None
        self.vit_model = None # Changed from cnn_model to vit_model
        self.previous_center_points = None
        self.vit_class_names = [] # Changed from cnn_class_names to vit_class_names

        if dataset_root_dir:
            try:
                temp_dataset = FormationDataset(root_dir=dataset_root_dir)
                self.vit_class_names = temp_dataset.idx_to_class # Changed
                print(f"ViT class names loaded from dataset: {self.vit_class_names}")
            except Exception as e:
                print(f"Error loading ViT class names from dataset {dataset_root_dir}: {e}")

        if yolo_model_path and os.path.exists(yolo_model_path):
            try:
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_model_path)
                print(f"YOLOv5 model loaded from {yolo_model_path}")
            except Exception as e:
                print(f"Error loading YOLOv5 model from {yolo_model_path}: {e}")
        else:
            print(f"YOLOv5 model path not provided or file not found: {yolo_model_path}. YOLOv5 will not be used.")

        if cnn_model_path and os.path.exists(cnn_model_path) and self.vit_class_names: # Changed cnn_model_path
            try:
                # Initialize ViT model with correct parameters
                self.vit_model = FormationClassifier(img_size=64, patch_size=8, in_channels=1, 
                                                     num_classes=len(self.vit_class_names), 
                                                     embed_dim=384, depth=6, num_heads=12, 
                                                     class_names=self.vit_class_names) # Corrected parameters
                self.vit_model.load_state_dict(torch.load(cnn_model_path)) # cnn_model_path now points to ViT model
                self.vit_model.eval()
                print(f"ViT model loaded from {cnn_model_path} with {len(self.vit_class_names)} classes: {self.vit_class_names}")
            except Exception as e:
                print(f"Error loading ViT model from {cnn_model_path}: {e}")
        else:
            print(f"ViT model path not provided, file not found, or class names not loaded. Formation classification will not be used.")

        # ViT input transform: Resize to 64x64, convert to Tensor (grayscale)
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) # Added normalization
        ])

    def _calculate_direction(self, current_center_points):
        direction_vector = {'x': 0.0, 'y': 0.0, 'z': 0.0}

        if self.previous_center_points is None or len(self.previous_center_points) == 0 or len(current_center_points) == 0:
            return direction_vector

        prev_avg_x = sum([p[0] for p in self.previous_center_points]) / len(self.previous_center_points)
        prev_avg_y = sum([p[1] for p in self.previous_center_points]) / len(self.previous_center_points)
        curr_avg_x = sum([p[0] for p in current_center_points]) / len(current_center_points)
        curr_avg_y = sum([p[1] for p in current_center_points]) / len(current_center_points)

        delta_x = curr_avg_x - prev_avg_x
        delta_y = curr_avg_y - prev_avg_y

        if abs(delta_x) < 5 and abs(delta_y) < 5:
            return direction_vector

        direction_vector['x'] = float(delta_x)
        direction_vector['y'] = float(delta_y)
        
        return direction_vector

    def _create_pattern_image(self, detected_objects, original_w, original_h, target_img_size=(64, 64)):
        # Create a black PIL Image
        img = Image.new('L', target_img_size, color=0) # Grayscale image, black background
        draw = ImageDraw.Draw(img)

        # Scaling factors to map original coordinates to the target image size
        scale_x = target_img_size[0] / original_w
        scale_y = target_img_size[1] / original_h

        center_points = []
        for obj in detected_objects:
            center_x = (obj.box['x'] + obj.box['width'] / 2) * scale_x
            center_y = (obj.box['y'] + obj.box['height'] / 2) * scale_y
            center_points.append((center_x, center_y))

        # Draw a white circle at each point
        for p in center_points:
            x, y = int(p[0]), int(p[1])
            # Ensure points are within bounds
            if 0 <= x < target_img_size[0] and 0 <= y < target_img_size[1]:
                draw.ellipse((x - POINT_RADIUS, y - POINT_RADIUS, x + POINT_RADIUS, y + POINT_RADIUS), fill=255) # White circle

        # Convert PIL Image to NumPy array for OpenCV processing
        img_array = np.array(img)

        # Apply Gaussian blur to create heatmap
        ksize = int(6 * GAUSSIAN_SIGMA + 1) # Ensure ksize is odd
        if ksize % 2 == 0: ksize += 1
        blurred_array = cv2.GaussianBlur(img_array, (ksize, ksize), GAUSSIAN_SIGMA)

        # Convert back to PIL Image (grayscale 'L' mode)
        img = Image.fromarray(blurred_array, mode='L')
        return img

    def analyze_image(self, image_path, debug_output_path=None):
        start_time = time.time()
        analysis_result = FlockAnalysisResult()
        
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return analysis_result

        cv_image = cv2.imread(image_path)
        if cv_image is None:
            print(f"Could not read image: {image_path}")
            return analysis_result

        h, w, _ = cv_image.shape
        analysis_result.image_width = w
        analysis_result.image_height = h

        if not self.yolo_model:
            print("YOLOv5 model not loaded. Skipping bird detection.")
            return analysis_result

        try:
            results = self.yolo_model(cv_image)
            df = results.pandas().xyxy[0]
            
            analysis_result.number_of_birds = len(df)
            if analysis_result.number_of_birds == 0:
                self.previous_center_points = None
                return analysis_result

            center_points, total_bird_area, species_set = [], 0, set()
            detected_objects_list = [] # To store DetectedObject instances

            for _, row in df.iterrows():
                obj = DetectedObject(
                    class_id=int(row['class']),
                    prob=float(row['confidence']),
                    x=int(row['xmin']),
                    y=int(row['ymin']),
                    width=int(row['xmax'] - row['xmin']),
                    height=int(row['ymax'] - row['ymin']),
                    species_name=row['name']
                )
                analysis_result.objects.append(obj)
                detected_objects_list.append(obj)
                
                center_points.append([obj.box['x'] + obj.box['width'] // 2, obj.box['y'] + obj.box['height'] // 2])
                total_bird_area += obj.box['width'] * obj.box['height']
                species_set.add(row['name'])

            analysis_result.species = list(species_set)
            analysis_result.is_flock = analysis_result.number_of_birds > 5

            if self.vit_model and analysis_result.number_of_birds > 1: # Changed to vit_model
                # Create pattern image from detected objects (bounding boxes)
                pattern_image = self._create_pattern_image(detected_objects_list, original_w=w, original_h=h)

                if debug_output_path:
                    pattern_image.save(debug_output_path) # Save the heatmap pattern image
                    print(f"Debug pattern image saved to {debug_output_path}")

                # Prepare image for ViT input
                input_tensor = self.transform(pattern_image).unsqueeze(0) # Add batch dimension

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                input_tensor = input_tensor.to(device)

                with torch.no_grad():
                    self.vit_model.to(device) # Changed to vit_model
                    outputs = self.vit_model(input_tensor) # Changed to vit_model

                    predicted_formation = self.vit_model.decode_prediction(outputs) # Changed to vit_model
                    analysis_result.formation = predicted_formation
                    print(f"Predicted Formation: {analysis_result.formation}")

                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    print("Class Probabilities:")
                    for i, prob in enumerate(probabilities):
                        class_name = self.vit_class_names[i] # Changed to vit_class_names
                        print(f"  {class_name}: {prob.item():.4f}")

            # Calculate critical points after formation is predicted
            analysis_result.critical_points = self._calculate_critical_points(analysis_result.formation, center_points, w, h, pattern_image, debug_output_path) # Pass pattern_image and debug_output_path

            if analysis_result.number_of_birds > 2:
                clustering = DBSCAN(eps=w*0.05, min_samples=3).fit(np.array(center_points)) # Adjusted eps for better clustering
                analysis_result.number_of_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            else:
                analysis_result.number_of_clusters = analysis_result.number_of_birds

            analysis_result.density = float(total_bird_area) / float(w * h)
            
            analysis_result.direction = self._calculate_direction(center_points)
            self.previous_center_points = center_points

        except Exception as e:
            print(f'Error in analyze_image: {e}')

        end_time = time.time()
        analysis_result.inference_time = end_time - start_time
        return analysis_result

    def _calculate_critical_points(self, formation, center_points, img_w, img_h, pattern_image=None, debug_output_path=None):
        critical_points = []
        if not center_points: # Handle empty center_points list
            return critical_points

        points_np = np.array(center_points)

        if formation == "V-shape":
            if pattern_image is not None:
                # Convert PIL Image to NumPy array for OpenCV processing
                pattern_img_np = np.array(pattern_image)

                # Binarize the pattern image based on brightness value 40
                _, binary_pattern_img = cv2.threshold(pattern_img_np, 40, 255, cv2.THRESH_BINARY)
                
                # Save binary image for debugging
                if debug_output_path:
                    binary_debug_path = debug_output_path.replace(".jpg", "_binary.png")
                    cv2.imwrite(binary_debug_path, binary_pattern_img)
                    print(f"Debug binary pattern image saved to {binary_debug_path}")

                # Ensure image is truly binary (0 or 1) for skeletonize
                binary_img_bool = binary_pattern_img > 0 

                # Perform skeletonization
                skeleton = skeletonize(binary_img_bool)
                skeleton_img = (skeleton * 255).astype(np.uint8)

                # Apply Canny Edge Detection on the skeleton
                edges = cv2.Canny(skeleton_img, 50, 150, apertureSize=3)

                # Hough Line Transform (Probabilistic Hough Line Transform)
                lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)

                intersection_points = []
                if lines is not None:
                    # Calculate intersections of line segments
                    for i in range(len(lines)):
                        for j in range(i + 1, len(lines)):
                            intersection = _find_line_intersection(lines[i], lines[j])
                            if intersection:
                                # Filter intersection points to be within image bounds
                                if 0 <= intersection[0] < skeleton_img.shape[1] and \
                                   0 <= intersection[1] < skeleton_img.shape[0]:
                                    intersection_points.append(intersection)

                # Identify the V-shape apex from intersection points
                apex_point = None
                if intersection_points:
                    # For a typical V-shape, the apex is the lowest point.
                    apex_point = min(intersection_points, key=lambda p: p[1])

                # Save skeleton image with unscaled apex for debugging
                if debug_output_path:
                    skeleton_debug_path = debug_output_path.replace(".jpg", "_skeleton.png")
                    skeleton_img_bgr = cv2.cvtColor(skeleton_img, cv2.COLOR_GRAY2BGR)
                    if apex_point:
                        cv2.circle(skeleton_img_bgr, (int(apex_point[0]), int(apex_point[1])), 3, (0, 0, 255), -1) # Red dot
                    cv2.imwrite(skeleton_debug_path, skeleton_img_bgr)
                    print(f"Debug skeleton image with apex saved to {skeleton_debug_path}")

                # Now, scale the apex point for the final result and add it to critical_points
                if apex_point:
                    # Scale coordinates from pattern size to original image size
                    pattern_size = (64, 64)
                    scale_x = img_w / pattern_size[0]
                    scale_y = img_h / pattern_size[1]
                    scaled_x = apex_point[0] * scale_x
                    scaled_y = apex_point[1] * scale_y
                    
                    scaled_apex = (scaled_x, scaled_y)
                    print(f"Identified V-shape apex by line intersection: {scaled_apex} (scaled from {apex_point})")
                    critical_points.append(scaled_apex)
                else:
                    print("No valid intersection points found for V-shape apex.")

        elif formation == "Line":
            # Find the two endpoints of the line
            # Calculate pairwise distances and find the two points with max distance
            if len(points_np) >= 2:
                distances = pdist(points_np)
                dist_matrix = squareform(distances)
                max_dist_indices = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
                critical_points.append(points_np[max_dist_indices[0]].tolist())
                critical_points.append(points_np[max_dist_indices[1]].tolist())
            elif len(points_np) == 1:
                critical_points.append(points_np[0].tolist())

        elif formation == "dispersed":
            # Find the center of the densest cluster
            # Use DBSCAN to find clusters, then find the largest cluster's centroid
            if len(points_np) >= 2:
                # Adjust eps based on image size or point density
                db = DBSCAN(eps=img_w * 0.05, min_samples=3).fit(points_np)
                labels = db.labels_
                unique_labels = set(labels)
                
                # Find the largest cluster (excluding noise if present)
                largest_cluster_label = -1
                max_cluster_size = 0
                for k in unique_labels:
                    if k == -1: # Noise points
                        continue
                    class_member_mask = (labels == k)
                    cluster_size = len(points_np[class_member_mask])
                    if cluster_size > max_cluster_size:
                        max_cluster_size = cluster_size
                        largest_cluster_label = k
                
                if largest_cluster_label != -1:
                    class_member_mask = (labels == largest_cluster_label)
                    cluster_points = points_np[class_member_mask]
                    centroid = np.mean(cluster_points, axis=0)
                    critical_points.append(centroid.tolist())
                else: # No significant clusters found, use overall centroid
                    centroid = np.mean(points_np, axis=0)
                    critical_points.append(centroid.tolist())
            elif len(points_np) == 1:
                critical_points.append(points_np[0].tolist())

        elif formation == "Ring":
            # Find the center of the ring (centroid of all points)
            centroid = np.mean(points_np, axis=0)
            critical_points.append(centroid.tolist())
        
        return critical_points

    def visualize_result(self, image_path, analysis_result, output_path="output.jpg"):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image for visualization: {image_path}")
            return

        # 바운딩 박스 그리기
        for obj in analysis_result.objects:
            x, y, w, h = obj.box['x'], obj.box['y'], obj.box['width'], obj.box['height']
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{obj.species} {obj.prob:.2f}"
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        #분석 결과 텍스트 추가
        info_text = [
            f"n_birds: {analysis_result.number_of_birds}",
            f"is_flock: {analysis_result.is_flock}",
            f"Species: {', '.join(analysis_result.species)}",
            f"formation: {analysis_result.formation}",
            f"density: {analysis_result.density:.2f}",
            f"n_clusters: {analysis_result.number_of_clusters}",
            f"direction: X={analysis_result.direction['x']:.2f}, Y={analysis_result.direction['y']:.2f}",
            f"Inference Time: {analysis_result.inference_time:.4f} s"
        ]
        
        # Add critical points to info text
        if analysis_result.critical_points:
            cp_text = ", ".join([f"({int(p[0])}, {int(p[1])})" for p in analysis_result.critical_points])
            info_text.append(f"Critical Points: {cp_text}")

        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(img, text, (10, y_offset + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Draw critical points on the image, snapped to the nearest bounding box center
        if analysis_result.critical_points and analysis_result.objects:
            bbox_centers = np.array([
                (obj.box['x'] + obj.box['width'] / 2, obj.box['y'] + obj.box['height'] / 2)
                for obj in analysis_result.objects
            ])

            for cp in analysis_result.critical_points:
                # Find the nearest bounding box center to the critical point
                distances = np.linalg.norm(bbox_centers - np.array(cp), axis=1)
                nearest_bbox_idx = np.argmin(distances)
                snapped_point = bbox_centers[nearest_bbox_idx]

                # Determine color based on formation
                color = (255, 0, 255)  # Default: Magenta
                if analysis_result.formation == "V-shape":
                    color = (0, 0, 255)  # Red for V-shape
                
                # Draw the circle at the snapped position
                cv2.circle(img, (int(snapped_point[0]), int(snapped_point[1])), 7, color, -1)

        cv2.imwrite(output_path, img)
        print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo Bird Flock Analyzer.")
    parser.add_argument("--image_dir", type=str,
                        default="visualization/sample",
                        help="Path to the directory containing images for analysis.")
    parser.add_argument("--output_dir", type=str,
                        default="visualization/result",
                        help="Path to the directory to save analysis results.")
    
    args = parser.parse_args()

    MODEL_DIR = "models"
    DATASET_DIR = "train_flock_analyzer/dataset"
    YOLO_MODEL_PATH = os.path.join(MODEL_DIR, "yolov5_best.pt")
    CNN_MODEL_PATH = os.path.join(MODEL_DIR, "analyzer_ViT.pth") # Changed model path to ViT

    os.makedirs(args.output_dir, exist_ok=True)

    analyzer = PseudoFlockAnalyzer(yolo_model_path=YOLO_MODEL_PATH, cnn_model_path=CNN_MODEL_PATH, dataset_root_dir=DATASET_DIR) # cnn_model_path now points to ViT model

    if os.path.exists(args.image_dir) and os.path.isdir(args.image_dir):
        image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print(f"No image files found in {args.image_dir}")
        
        for image_file in image_files:
            image_path = os.path.join(args.image_dir, image_file)
            print(f"\nAnalyzing image: {image_path}")
            debug_output_dir = "visualization/debug"
            os.makedirs(debug_output_dir, exist_ok=True)
            debug_output_path = os.path.join(debug_output_dir, f"pattern_{image_file}")
            analysis_result = analyzer.analyze_image(image_path, debug_output_path=debug_output_path)
            print("--- Analysis Result ---")
            print(f"Number of Birds: {analysis_result.number_of_birds}")
            print(f"Is Flock: {analysis_result.is_flock}")
            print(f"Species: {', '.join(analysis_result.species)}")
            print(f"Formation: {analysis_result.formation}")
            print(f"Density: {analysis_result.density:.2f}")
            print(f"Clusters: {analysis_result.number_of_clusters}")
            print(f"Direction: X={analysis_result.direction['x']:.2f}, Y={analysis_result.direction['y']:.2f}")
            print(f"Inference Time: {analysis_result.inference_time:.4f} s")
            
            output_image_path = os.path.join(args.output_dir, f"result_{image_file}")
            analyzer.visualize_result(image_path, analysis_result, output_image_path)
    else:
        print(f"Image directory not found or is not a directory: {args.image_dir}")