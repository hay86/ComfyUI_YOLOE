import os
import torch
import supervision as sv
import numpy as np
import json

from ultralytics import YOLOE
from torchvision.transforms.v2 import ToTensor, ToPILImage

to_tensor = ToTensor()
to_image = ToPILImage()

class YOLOENode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    'model': ('YOLOEMODEL',),
                    'images': ('IMAGE',),
                    'confidence_threshold': ('FLOAT', {'default': 0.25, 'min': 0, 'max': 1, 'step': 0.01}),
                    'iou_threshold': ('FLOAT', {'default': 0.7, 'min': 0, 'max': 1, 'step': 0.01}),
                    'image_size': ('INT', {'default': 640, 'min': 320, 'max': 1024, 'step': 32}),
                     },}

    CATEGORY = "YOLOENode"

    RETURN_TYPES = ("DETECTIONS", )
    RETURN_NAMES = ("detections", )
    FUNCTION = "inference"


    def inference(self, model, images, confidence_threshold, iou_threshold, image_size):
        output = []
        images = images.permute([0,3,1,2])

        for img in images:
            image = to_image(img)
            results = model.predict(source=image, imgsz=image_size, conf=confidence_threshold, iou=iou_threshold)
            detections = sv.Detections.from_ultralytics(results[0])
            output.append(detections)
        return [output]

class YOLOENode_ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        default_categories = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        return {"required": {
                    'model_id': ([
                        'yoloe-v8s-seg', 
                        'yoloe-v8m-seg',
                        'yoloe-v8l-seg',
                        'yoloe-11s-seg', 
                        'yoloe-11m-seg',
                        'yoloe-11l-seg',], {'default': 'yoloe-v8l-seg'}),
                    'categories': (
                        'STRING', {
                            'display': 'Categories',
                            'default': ','.join(default_categories),
                            'multiline': True})
                     },}

    CATEGORY = "YOLOENode"

    RETURN_TYPES = ("YOLOEMODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "load_model"


    def load_model(self, model_id, categories):
        model = YOLOE.from_pretrained(f"jameslahm/{model_id}")
        model.to("cuda")
        model.eval()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        mobileclip_path = os.path.join(current_dir, "mobileclip_blt.pt")

        if not os.path.exists(mobileclip_path):
            download_url = "https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/mobileclip_blt.pt"
            os.system(f"wget {download_url} -O {mobileclip_path}")

        names = [category.strip() for category in categories.split(',')]
        model.set_classes(names, model.get_text_pe(names))

        return (model,)

class YOLOENode_Display:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    'images': ('IMAGE',),
                    'detections': ('DETECTIONS',),
                     },}

    CATEGORY = "YOLOENode"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "display"


    def display(self, images, detections):
        output = []
        images = images.permute([0,3,1,2])

        for img, detection in zip(images, detections):
            image = to_image(img)

            resolution_wh = image.size
            thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
            text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

            labels = [
                f"{class_name} {confidence:.2f}"
                for class_name, confidence in zip(detection.data["class_name"], detection.confidence)
            ]

            annotated_image = image.copy()
            annotated_image = sv.MaskAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                opacity=0.4
            ).annotate(scene=annotated_image, detections=detection)
            annotated_image = sv.BoxAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                thickness=thickness
            ).annotate(scene=annotated_image, detections=detection)
            annotated_image = sv.LabelAnnotator(
                color_lookup=sv.ColorLookup.INDEX,
                text_scale=text_scale,
                smart_position=True
            ).annotate(scene=annotated_image, detections=detection, labels=labels)

            output.append(to_tensor(annotated_image))

        output = torch.stack(output, dim=0)
        output = output.permute([0,2,3,1])
                
        return (output[:, :, :, :3],)
    

class YOLOENode_Display_JSON:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    'detections': ('DETECTIONS',),
                    'max_conf_only': ('BOOLEAN', {'default': True}),
                    'num_mask_samples': ('INT', {'default': 10, 'min': 1, 'max': 100, 'step': 1}),
                     },}

    CATEGORY = "YOLOENode"

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("JSON", )
    FUNCTION = "display"


    def display(self, detections, max_conf_only, num_mask_samples):
        output = []

        for detection in detections:
            if max_conf_only:
                max_per_class = {}  # class_id -> (idx, conf)
                for idx, (cls_id, conf) in enumerate(zip(detection.class_id, detection.confidence)):
                    if cls_id not in max_per_class or conf > max_per_class[cls_id][1]:
                        max_per_class[cls_id] = (idx, conf)
                top_conf_list = list(max_per_class.values())
            else:
                top_conf_list = [(idx, conf) for idx, conf in enumerate(detection.confidence)]

            top_conf_list.sort(key=lambda x: x[1], reverse=True)

            for idx, conf in top_conf_list:
                cls_id = detection.class_id[idx]
                cls_name = detection.data["class_name"][idx]
                xyxy = detection.xyxy[idx].tolist()

                mask = detection.mask[idx]
                y, x = np.nonzero(mask)
                sample_indices = np.random.choice(len(x), size=min(len(x), num_mask_samples), replace=False)
                sampled_points = np.stack([x[sample_indices], y[sample_indices]], axis=1)
                mask_samples = sampled_points.tolist()

                output.append({
                    "class_id": int(cls_id),
                    "class_name": str(cls_name),
                    "xyxy": xyxy,
                    "mask_samples": mask_samples,
                    "confidence": float(conf)
                })
            print(output)
                
        return (json.dumps(output),)


NODE_CLASS_MAPPINGS = {
    "D_YOLOENode": YOLOENode,
    "D_YOLOENode_ModelLoader": YOLOENode_ModelLoader,
    "D_YOLOENode_Display": YOLOENode_Display,
    "D_YOLOENode_Display_JSON": YOLOENode_Display_JSON,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_YOLOENode": "YOLOE Node",
    "D_YOLOENode_ModelLoader": "YOLOE Model Loader",
    "D_YOLOENode_Display": "YOLOE Display",
    "D_YOLOENode_Display_JSON": "YOLOE Display JSON",
}