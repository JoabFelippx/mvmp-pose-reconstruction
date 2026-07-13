from ultralytics import YOLO
import cv2
import numpy as np

from is_msgs.image_pb2 import ObjectAnnotations
import torch

class SkeletonsDetector:

    def __init__(self, yolo_model_path):
        self._model = YOLO(yolo_model_path)
        dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
        for _ in range(5):
            self._model(dummy_image, device="cuda:0", verbose=False)
        
        self.cuda_stream = torch.cuda.Stream(device="cuda:0")

    def to_object_annotations(self, humans, kp_scores, image_shape):
        
        obs = ObjectAnnotations()
        bboxes_xyxy = humans.boxes.xyxy.cpu().numpy().astype('uint32')
        kp_scores_list = kp_scores.flatten().tolist()
        
        i = 0
        for bboxe_xyxy in bboxes_xyxy:
            obj = obs.objects.add()
            vertex_1 = obj.region.vertices.add()
            vertex_1.x = bboxe_xyxy[0]
            vertex_1.y = bboxe_xyxy[1]
            vertex_2 = obj.region.vertices.add()
            vertex_2.x = bboxe_xyxy[2]
            vertex_2.y = bboxe_xyxy[3]

            bbox_keypoints = humans.keypoints.data.cpu().numpy().astype('uint32')[i]
            for kpt_id in range(len(bbox_keypoints)):
                part = obj.keypoints.add()
                part.id = kpt_id + 1
                part.position.x = bbox_keypoints[kpt_id][0]
                part.position.y = bbox_keypoints[kpt_id][1]
                part.score = kp_scores_list[kpt_id]
            try:
                obj.id = i
                 
            except:
                continue
            obj.label = 'human'
            obj.score = humans.boxes.conf.cpu().numpy().astype('float32')[i]

            i+= 1
            
        obs.resolution.width = image_shape[1]
        obs.resolution.height = image_shape[0]

        return obs
        
    def to_object_annotation_3d(self, humans, image_shape=(1280, 720)):
        
        obs = ObjectAnnotations()
        
        for skeleton in humans:
            obj = obs.objects.add()

            for kpt_id, kpt in skeleton['skeleton_3d'].items():
                
                part = obj.keypoints.add()
                part.id = int(kpt_id)
                part.position.x = kpt[0]
                part.position.y = kpt[1]
                part.position.z = kpt[2]
                part.score = 1.0

            obj.label = 'human_3d'
            obj.score = 1.0
            obj.id = int(skeleton.get('id')) + 1 
            
        return obs
    
    def detect(self, images):        
        results = []
        torch.cuda.synchronize()  # Sincronizar uma vez no início
        for img in images:
            result = self._model(img, device='cuda:0', verbose=False, iou=0.7)
            results.append(result[0])        
        torch.cuda.synchronize()  # Sincronizar uma vez no final
        return results
