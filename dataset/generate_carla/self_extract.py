
#!/usr/bin/env python3
"""
Autopilot Pedestrian Capture in CARLA using Semantic Segmentation

This script spawns a vehicle in autopilot mode and attaches both an RGB camera and a semantic segmentation camera.
Every N simulation ticks (e.g., every 20 ticks) the script captures an RGB image and its corresponding semantic segmentation image,
saves the segmentation image (for debugging) so you can inspect its labels, processes the segmentation image to extract connected regions
corresponding to pedestrians, computes 2D bounding boxes (in normalized YOLO format) for these pedestrians, and saves the RGB image,
a label text file (empty if no pedestrian is detected), and an annotated image.

Usage:
    python self_extract.py --host <CARLA_HOST> --port <PORT> --output_dir <output_directory> --max_images <num> [--start_index <n>]
"""

import os
import sys
import carla
import weakref
import random
import time
import numpy as np
import cv2
import argparse

# Camera settings
VIEW_WIDTH = 960
VIEW_HEIGHT = 540
VIEW_FOV = 90

# Define the expected pedestrian color in the CityScapes palette.
# Default: Pedestrians are painted as RGB (220, 20, 60).
PEDESTRIAN_COLOR = np.array([4, 0, 0], dtype=np.uint8)

# Minimum bounding box area threshold (in pixel^2 units)
MIN_BBOX_THRESHOLD = 200

# ==============================================================================
# -- AutonomousPedestrianCapture Class ----------------------------------------
# ==============================================================================

class AutonomousPedestrianCapture:
    def __init__(self, host, port, output_dir, max_images, start_index=0):
        self.host = host
        self.port = port
        self.output_dir = output_dir
        # max_images now means the number of images to capture in this run.
        self.max_images = max_images
        self.start_index = start_index
        self.image_count = 0  # Counter for images captured in this run

        self.client = None
        self.world = None
        self.vehicle = None
        self.rgb_camera = None
        self.seg_camera = None
        self.last_rgb_image = None
        self.last_seg_image = None  # Will store the RGB version of the segmentation image

        # Create directories for saving data
        self.rgb_dir = os.path.join(output_dir, "RGB")
        self.label_dir = os.path.join(output_dir, "labels")
        self.annotated_dir = os.path.join(output_dir, "annotated")
        self.seg_dir = os.path.join(output_dir, "segmentation")
        for d in [self.rgb_dir, self.label_dir, self.annotated_dir, self.seg_dir]:
            os.makedirs(d, exist_ok=True)

    def setup(self):
        print("Setting up client...")
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.spawn_vehicle()
        self.setup_cameras()

    def spawn_vehicle(self):
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = random.choice(bp_lib.filter('vehicle.*'))
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle.set_autopilot(True)
        print("Spawned vehicle (id: {}) in autopilot mode.".format(self.vehicle.id))

    def setup_cameras(self):
        bp_lib = self.world.get_blueprint_library()

        # Setup RGB camera
        rgb_bp = bp_lib.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        rgb_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        rgb_bp.set_attribute('fov', str(VIEW_FOV))
        
        cam_location = carla.Location(x=0.0, y=0.0, z=2.5)
        cam_rotation = carla.Rotation(pitch=-10)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        self.rgb_camera = self.world.spawn_actor(rgb_bp, cam_transform, attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        self.rgb_camera.listen(lambda image: weak_self().process_rgb(image))
        print("RGB camera set up.")

        # Setup Semantic Segmentation camera
        seg_bp = bp_lib.find('sensor.camera.semantic_segmentation')
        seg_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        seg_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        seg_bp.set_attribute('fov', str(VIEW_FOV))
        self.seg_camera = self.world.spawn_actor(seg_bp, cam_transform, attach_to=self.vehicle)
        self.seg_camera.listen(lambda image: weak_self().process_segmentation(image))
        print("Semantic segmentation camera set up.")

    def process_rgb(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
        self.last_rgb_image = array.copy()

    def process_segmentation(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
        seg_rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        self.last_seg_image = seg_rgb.copy()

    def capture_pedestrian_data(self):
        if self.last_rgb_image is None or self.last_seg_image is None:
            print("RGB or segmentation image not available yet.")
            return

        rgb_image = self.last_rgb_image.copy()
        seg_image = self.last_seg_image.copy()

        pedestrian_mask = cv2.inRange(seg_image, PEDESTRIAN_COLOR, PEDESTRIAN_COLOR)

        if cv2.countNonZero(pedestrian_mask) == 0:
            print("Skipping frame: no pedestrians detected in segmentation mask.")
            return

                # get connected components
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(
            pedestrian_mask, connectivity=8)

        # 1) collect all raw pixel‐coordinate boxes
        raw_boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if w < 5 or h < 5 or area < 50:
                continue
            raw_boxes.append((x, y, x + w, y + h))

        if not raw_boxes:
            print("Skipping frame: no valid pedestrian regions.")
            return

        # 2) merge any overlapping boxes
        merged_boxes = []
        for (l, t, r, b) in raw_boxes:
            merged = False
            for idx, (ml, mt, mr, mb) in enumerate(merged_boxes):
                # if overlap
                if not (r < ml or l > mr or b < mt or t > mb):
                    # replace with union
                    merged_boxes[idx] = (
                        min(l, ml), min(t, mt),
                        max(r, mr), max(b, mb)
                    )
                    merged = True
                    break
            if not merged:
                merged_boxes.append((l, t, r, b))

        # 3) filter merged boxes by area threshold and build labels
        labels_list = []
        draw_boxes = []
        for (l, t, r, b) in merged_boxes:
            w_pix, h_pix = r - l, b - t
            if w_pix * h_pix < MIN_BBOX_THRESHOLD:
                continue
            cx = l + w_pix / 2.0
            cy = t + h_pix / 2.0
            norm_bbox = (
                cx / VIEW_WIDTH,
                cy / VIEW_HEIGHT,
                w_pix / VIEW_WIDTH,
                h_pix / VIEW_HEIGHT
            )
            labels_list.append(
                f"0 {norm_bbox[0]:.6f} {norm_bbox[1]:.6f} "
                f"{norm_bbox[2]:.6f} {norm_bbox[3]:.6f}"
            )
            draw_boxes.append(norm_bbox)

        if not labels_list:
            print(f"Skipping frame: no merged bbox above area threshold {MIN_BBOX_THRESHOLD}.")
            return


        # Use final_index = start_index + current counter for file naming.
        final_index = self.start_index + self.image_count

        seg_filename = os.path.join(self.seg_dir, f"seg_{final_index:05d}.png")
        cv2.imwrite(seg_filename, cv2.cvtColor(seg_image, cv2.COLOR_RGB2BGR))
        print(f"Segmentation image saved as: {seg_filename}")

        rgb_filename = os.path.join(self.rgb_dir, f"rgb_{final_index:05d}.png")
        cv2.imwrite(rgb_filename, rgb_image)

        label_filename = os.path.join(self.label_dir, f"rgb_{final_index:05d}.txt")
        with open(label_filename, "w") as f:
            for line in labels_list:
                f.write(line + "\n")

        annotated = rgb_image.copy()
        for bbox in draw_boxes:
            x_center, y_center, w_norm, h_norm = bbox
            left = int((x_center - w_norm / 2) * VIEW_WIDTH)
            right = int((x_center + w_norm / 2) * VIEW_WIDTH)
            top = int((y_center - h_norm / 2) * VIEW_HEIGHT)
            bottom = int((y_center + h_norm / 2) * VIEW_HEIGHT)
            color = (0, 0, 255)
            cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
            cv2.putText(annotated, "0", (left, top - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        annotated_filename = os.path.join(self.annotated_dir, f"annotated_{final_index:05d}.png")
        cv2.imwrite(annotated_filename, annotated)
        # compute the maximum pedestrian bbox area (in pixels)
        pixel_areas = [
            (bbox[2] * VIEW_WIDTH) * (bbox[3] * VIEW_HEIGHT)
            for bbox in draw_boxes
        ]
        max_area = max(pixel_areas) if pixel_areas else 0
        
        print(f"Captured frame {final_index:05d} with {len(labels_list)} pedestrian labels (max area: {max_area}).")

        self.image_count += 1

    def run(self):
        self.setup()
        tick = 0
        capture_interval = 20  # capture every 20 ticks
        try:
            while self.image_count < self.max_images:
                self.world.tick()
                tick += 1
                time.sleep(0.05)
                if tick % capture_interval == 0:
                    self.capture_pedestrian_data()
        except KeyboardInterrupt:
            print("Capture interrupted. Cleaning up...")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.rgb_camera:
            self.rgb_camera.destroy()
        if self.seg_camera:
            self.seg_camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        print("Cleanup complete. Exiting.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autopilot Pedestrian Capture in CARLA using Semantic Segmentation.")
    parser.add_argument("--host", default="192.168.32.1", help="CARLA host IP")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--output_dir", type=str, default="output_data_x1/", help="Directory to save images and labels")
    parser.add_argument("--max_images", type=int, default=20, help="Maximum number of images to capture in this run")
    parser.add_argument("--start_index", type=int, default=0, help="Starting index for image numbering")
    args = parser.parse_args()

    capture_client = AutonomousPedestrianCapture(args.host, args.port, args.output_dir, args.max_images, args.start_index)
    capture_client.run()
