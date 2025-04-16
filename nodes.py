import os
import torch
import supervision as sv

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


NODE_CLASS_MAPPINGS = {
    "D_YOLOENode": YOLOENode,
    "D_YOLOENode_ModelLoader": YOLOENode_ModelLoader,
    "D_YOLOENode_Display": YOLOENode_Display,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "D_YOLOENode": "YOLOE Node",
    "D_YOLOENode_ModelLoader": "YOLOE Model Loader",
    "D_YOLOENode_Display": "YOLOE Display",
}