import json

def create_coco_dict(rf_categories):
    """
    Create a COCO format dictionary with rf_categories from roboflow dataset
    """
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": rf_categories
    }


def add_to_coco(coco_dict, rf_categories, boxes, labels, h, w, img_name):
    """
    :param coco_dict: dict, COCO format dictionary to update
    :param rf_categories: list of coco categories, each category is a dict with 'id' and 'name'
    :param boxes: list of bounding boxes, each box is [x1, y1, x2, y2]
    :param labels: list of labels corresponding to each box
    :param h: int, height of the image
    :param w: int, width of the image
    :return: None, modifies coco_dict in place
    """

    # Ensure coco_dict has the required structure
    image_id = len(coco_dict["images"])
    coco_dict["images"].append({
        "id": image_id,
        "file_name": img_name,
        "width": w,
        "height": h
    })
    # Update annotations
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        annotation = {
            "id": len(coco_dict["annotations"]),
            "image_id": image_id,
            "category_id": label,
            "bbox": [x1, y1, width, height],
            "area": width * height,
            "iscrowd": 0
        }
        coco_dict["annotations"].append(annotation)
    
    