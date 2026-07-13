import numpy as np


class Skeleton():
    def __init__(self, skeleton_obj, id, camera_id):
        self.id = id
        self.skeleton_obj = skeleton_obj
        self.camera_id = camera_id


class Reconstructor3D:

    def __init__(self, projection_matrices, num_cameras, num_keypoints, dist_threshold=0.25):
        self.projection_matrices = projection_matrices
        self.num_cameras         = num_cameras
        self.num_keypoints       = num_keypoints
        self.dist_threshold      = dist_threshold

    def _create_skeleton_objects_for_person(self, person_match, all_annotations):
        skeleton_objects = []
        for cam_idx, sk_id in person_match.items():
            if cam_idx < len(all_annotations):
                for sk_obj in all_annotations[cam_idx].objects:
                    if sk_id == sk_obj.id:
                        skeleton_objects.append(Skeleton(sk_obj, sk_id, camera_id=cam_idx + 1))
                        break
        return skeleton_objects

    def _to_3d_keypoints_structure(self, skeleton_objs):
        sk_points = np.zeros((self.num_cameras, self.num_keypoints, 2))
        for sk_obj in skeleton_objs:
            cam_idx = sk_obj.camera_id - 1
            for kp in sk_obj.skeleton_obj.keypoints:
                if kp.id < self.num_keypoints:
                    sk_points[cam_idx, kp.id] = [kp.position.x, kp.position.y]

        valid_keypoints_info = {}
        for kp_idx in range(self.num_keypoints):
            cameras_with_point = []
            for cam_idx in range(self.num_cameras):
                point = sk_points[cam_idx, kp_idx]
                if not np.allclose(point, [0, 0]):
                    cameras_with_point.append((cam_idx + 1, point))
            if len(cameras_with_point) >= 2:
                valid_keypoints_info[kp_idx] = cameras_with_point

        return valid_keypoints_info

    def _reconstruct_points_from_svd(self, valid_keypoints_info):
        dots_3d_all = {}
        for kp_idx, cam_data in valid_keypoints_info.items():
            num_cams = len(cam_data)
            A = np.zeros((2 * num_cams, 4))
            for i, (cam_id, point2d) in enumerate(cam_data):
                P = self.projection_matrices[cam_id - 1]
                A[2 * i]     = point2d[0] * P[2, :] - P[0, :]
                A[2 * i + 1] = point2d[1] * P[2, :] - P[1, :]

            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, 0:4]

            if X[3] != 0:
                X = X / X[3]
                dots_3d_all[kp_idx] = [X[0], X[1], X[2]]

        return dots_3d_all

    def _merge_duplicates(self, persons, all_annotations):
        """
        Recebe lista de dicts internos {skeleton_3d, matche_2d, average_point},
        funde os que estão abaixo de dist_threshold e retorna apenas skeleton_3d crus.
        """
        m = len(persons)
        if m <= 1:
            return [p["skeleton_3d"] for p in persons]

        to_remove = set()

        for i in range(m):
            if i in to_remove:
                continue
            for j in range(i + 1, m):
                if j in to_remove:
                    continue

                dist = np.linalg.norm(
                    persons[i]["average_point"] - persons[j]["average_point"]
                )

                if dist < self.dist_threshold:
                    merged_matches = persons[i]["matche_2d"].copy()
                    merged_matches.update(persons[j]["matche_2d"])

                    sk_objs   = self._create_skeleton_objects_for_person(merged_matches, all_annotations)
                    valid_kps = self._to_3d_keypoints_structure(sk_objs)
                    new_3d    = self._reconstruct_points_from_svd(valid_kps)

                    if new_3d:
                        persons[i]["skeleton_3d"]   = new_3d
                        persons[i]["matche_2d"]     = merged_matches
                        persons[i]["average_point"] = np.mean(
                            np.array(list(new_3d.values())), axis=0
                        )

                    to_remove.add(j)

        return [p["skeleton_3d"] for idx, p in enumerate(persons) if idx not in to_remove]

    def reconstruct_all(self, matched_persons, all_annotations):
        """
        Mesma interface de sempre — retorna lista de skeleton_3d crus.
        Internamente funde esqueletos duplicados antes de retornar.
        """
        persons = []

        for person_data in matched_persons:
            person_match = person_data['ids']
            sk_objs      = self._create_skeleton_objects_for_person(person_match, all_annotations)
            valid_kps    = self._to_3d_keypoints_structure(sk_objs)
            skeleton_3d  = self._reconstruct_points_from_svd(valid_kps)

            if skeleton_3d:
                average_point = np.mean(np.array(list(skeleton_3d.values())), axis=0)
                persons.append({
                    "skeleton_3d":   skeleton_3d,
                    "matche_2d":     person_match,
                    "average_point": average_point,
                })

        return self._merge_duplicates(persons, all_annotations)
import numpy as np

class Skeleton():
    def __init__(self, skeleton_obj, id, camera_id):
        
        self.id = id
        self.skeleton_obj = skeleton_obj
        self.camera_id = camera_id

class Reconstructor3D:
    
    def __init__(self, projection_matrices, num_cameras, num_keypoints):
        self.projection_matrices = projection_matrices
        self.num_cameras = num_cameras
        self.num_keypoints = num_keypoints
        
    def _create_skeleton_objects_for_person(self, person_match, all_annotations):
        skeleton_objects = []
        for cam_idx, sk_id in person_match.items():
            if cam_idx < len(all_annotations):
                for sk_obj in all_annotations[cam_idx].objects:
                    if sk_id == sk_obj.id:
                        skeleton_objects.append(Skeleton(sk_obj, sk_id, camera_id=cam_idx + 1))
                        break
                        
        return skeleton_objects
        
    def _to_3d_keypoints_structure(self, skeleton_objs):
        
        sk_points = np.zeros((self.num_cameras, self.num_keypoints, 2))
        for sk_obj in skeleton_objs:
            cam_idx = sk_obj.camera_id - 1
            for kp in sk_obj.skeleton_obj.keypoints:
                if kp.id < self.num_keypoints:
                    sk_points[cam_idx, kp.id] = [kp.position.x, kp.position.y]
        
        valid_keypoints_info = {}
        for kp_idx in range(self.num_keypoints):
            cameras_with_point = []
            for cam_idx in range(self.num_cameras):
                point = sk_points[cam_idx, kp_idx]
                if not np.allclose(point, [0, 0]):
                    cameras_with_point.append((cam_idx, point))
                    
            if len(cameras_with_point) >= 2:
                valid_keypoints_info[kp_idx] = cameras_with_point
                
        return valid_keypoints_info
        
    def _reconstruct_points_from_svd(self, valid_keypoints_info):
        
        dots_3d_all = {}
        for kp_idx, cam_data in valid_keypoints_info.items():
            num_cams = len(cam_data)
            A = np.zeros((2 * num_cams, 4))
            for i, (cam_id, point2d) in enumerate(cam_data):
                P = self.projection_matrices[cam_id]
                A[2*i]   = point2d[0] * P[2,:] - P[0,:]
                A[2*i+1] = point2d[1] * P[2,:] - P[1,:]

            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1, 0:4]
            
            if X[3] != 0:
                X = X / X[3]
                dots_3d_all[kp_idx] = [X[0]  , X[1]  , X[2]  ]
                # dots_3d_all[kp_idx] = [X[2] / 100, X[0] / 100, -X[1] / 100]
        return dots_3d_all
        
        
    def reconstruct_all(self, matched_persons, all_annotations):
        
        
        reconstructed_skeletons = []

        for person_data in matched_persons:
            person_match = person_data['ids']

            skeleton_objects = self._create_skeleton_objects_for_person(person_match, all_annotations)
            valid_kps = self._to_3d_keypoints_structure(skeleton_objects)
            person_3d = self._reconstruct_points_from_svd(valid_kps)

            if person_3d:
                reconstructed_skeletons.append(person_3d)

        return reconstructed_skeletons


# import numpy as np
# from itertools import combinations

# class Skeleton():
#     def __init__(self, skeleton_obj, id, camera_id):
#         self.id = id
#         self.skeleton_obj = skeleton_obj
#         self.camera_id = camera_id


# class Reconstructor3D:

#     def __init__(self, projection_matrices, num_cameras, num_keypoints, all_calibs_parameters, dist_threshold=25):
#         self.projection_matrices = projection_matrices
#         self.num_cameras         = num_cameras
#         self.num_keypoints       = num_keypoints
#         self.dist_threshold      = dist_threshold
#         self.all_calibs_parameters = all_calibs_parameters

#     def _create_skeleton_objects_for_person(self, person_match, all_annotations):
#         skeleton_objects = []
#         for cam_idx, sk_id in person_match.items():
#             if cam_idx < len(all_annotations):
#                 for sk_obj in all_annotations[cam_idx].objects:
#                     if sk_id == sk_obj.id:
#                         skeleton_objects.append(Skeleton(sk_obj, sk_id, camera_id=cam_idx + 1))
#                         break
#         return skeleton_objects


#     def _to_3d_keypoints_structure(self, skeleton_objs):
#         sk_points = np.zeros((self.num_cameras, self.num_keypoints, 2))
#         for sk_obj in skeleton_objs:
#             cam_idx = sk_obj.camera_id - 1
#             for kp in sk_obj.skeleton_obj.keypoints:
#                 if kp.id < self.num_keypoints:
#                     sk_points[cam_idx, kp.id] = [kp.position.x, kp.position.y]

#         valid_keypoints_info = {}
#         for kp_idx in range(self.num_keypoints):
#             cameras_with_point = []
#             for cam_idx in range(self.num_cameras):
#                 point = sk_points[cam_idx, kp_idx]
#                 if not np.allclose(point, [0, 0]):
#                     cameras_with_point.append((cam_idx + 1, point))
#             if len(cameras_with_point) >= 2:
#                 valid_keypoints_info[kp_idx] = cameras_with_point

#         return valid_keypoints_info

#     def _reconstruct_points_from_svd(self, valid_keypoints_info):
#         dots_3d_all = {}
#         keypoints_matrix = np.zeros((18, 3))
        
#         for kp_idx, cam_data in valid_keypoints_info.items():
#             num_cams = len(cam_data)
#             A = np.zeros((2 * num_cams, 4))
#             for i, (cam_id, point2d) in enumerate(cam_data):
#                 P = self.projection_matrices[cam_id - 1]
#                 A[2 * i]     = point2d[0] * P[2, :] - P[0, :]
#                 A[2 * i + 1] = point2d[1] * P[2, :] - P[1, :]

#             _, _, Vt = np.linalg.svd(A)
#             X = Vt[-1, 0:4]

#             if X[3] != 0:
#                 X = X / X[3]
#                 keypoints_matrix[kp_idx] = [X[0], X[1], X[2]]
#         return keypoints_matrix

#     def _merge_duplicates(self, persons, all_annotations):
#         """
#         Recebe lista de dicts internos {skeleton_3d, matche_2d, average_point},
#         funde os que estão abaixo de dist_threshold e retorna apenas skeleton_3d crus.
#         """
#         m = len(persons)
#         if m <= 1:
#             return [p["skeleton_3d"] for p in persons]

#         to_remove = set()

#         for i in range(m):
#             if i in to_remove:
#                 continue
#             for j in range(i + 1, m):
#                 if j in to_remove:
#                     continue

#                 dist = np.linalg.norm(
#                     persons[i]["average_point"] - persons[j]["average_point"]
#                 )

#                 if dist < self.dist_threshold:
#                     merged_matches = persons[i]["matche_2d"].copy()
#                     merged_matches.update(persons[j]["matche_2d"])

#                     sk_objs   = self._create_skeleton_objects_for_person(merged_matches, all_annotations)
#                     valid_kps = self._to_3d_keypoints_structure(sk_objs)
#                     new_3d    = self._reconstruct_points_from_svd(valid_kps)
    
#                     non_zero = new_3d[~np.all(new_3d == 0, axis=1)]
#                     if len(non_zero) > 0:
#                         persons[i]["skeleton_3d"]   = new_3d
#                         persons[i]["matche_2d"]     = merged_matches
#                         persons[i]["average_point"] = np.mean(non_zero, axis=0)

#                     to_remove.add(j)

#         return [p["skeleton_3d"] for idx, p in enumerate(persons) if idx not in to_remove]


#     def generate_3d_candidates(self, all_annotations, matched_persons):
       
#         combs = combinations(list(matched_persons.keys()), 2)
#         new_mtch_persons = []
#         for comb in combs:
#             new_matched_person = {
#                 comb[0]: matched_persons[comb[0]],
#                 comb[1]: matched_persons[comb[1]]
#             }
#             new_mtch_persons.append(new_matched_person)

#         return new_mtch_persons, combs

#     def _reproject_skeleton(self, skeleton_3d, cam_ids):
#         """
#         Projects each keypoint in skeleton_3d (shape: num_keypoints x 3) into
#         each camera in cam_ids.

#         Returns
#         -------
#         reprojections : dict {cam_id: np.ndarray of shape (num_keypoints, 2)}
#         """
#         reprojections = {}

#         for cam_id in cam_ids:
#             R  = self.all_calibs_parameters[cam_id]["R"].reshape(3, 3)
#             t  = self.all_calibs_parameters[cam_id]["T"].reshape(3)
#             nK = self.all_calibs_parameters[cam_id]["nK"].reshape(3, 3)

#             points_2d = np.zeros((self.num_keypoints, 2))

#             for kp_idx in range(self.num_keypoints):
#                 X = skeleton_3d[kp_idx]

#                 if np.allclose(X, 0):
#                     continue

#                 Xc = R @ X + t

#                 if Xc[2] <= 0:
#                     continue

#                 x = Xc[0] / Xc[2]
#                 y = Xc[1] / Xc[2]

#                 proj = nK @ np.array([x, y, 1.0])
#                 points_2d[kp_idx] = [proj[0], proj[1]]

#             reprojections[cam_id] = points_2d

#         return reprojections

#     def _compute_reprojection_error(self, skeleton_3d, person_match, all_annotations):
#         """
#         Computes the mean reprojection error (px) for skeleton_3d against
#         the 2D detections in person_match.

#         Returns
#         -------
#         mean_error : float  (inf if no valid points found)
#         """
#         cam_ids = list(person_match.keys())
#         reproj  = self._reproject_skeleton(skeleton_3d, cam_ids)

#         total_error  = 0.0
#         total_points = 0

#         for cam_id, skt_id in person_match.items():

#             if cam_id >= len(all_annotations):
#                 continue

#             sk_obj = None
#             for obj in all_annotations[cam_id].objects:
#                 if obj.id == skt_id:
#                     sk_obj = obj
#                     break

#             if sk_obj is None:
#                 continue

#             for kp in sk_obj.keypoints:
#                 kp_id = kp.id
#                 if kp_id >= self.num_keypoints:
#                     continue

#                 kp_reproj   = reproj[cam_id][kp_id]
#                 original_2d = np.array([kp.position.x, kp.position.y])

#                 total_error  += np.linalg.norm(original_2d - kp_reproj)
#                 total_points += 1

#         if total_points == 0:
#             return float("inf")

#         return total_error / total_points

#     def reconstruct_all(self, matched_persons, all_annotations, reproj_threshold=20):
#         """
#         For each matched person:
#           1. Generates all pairwise 3D candidates.
#           2. Keeps only candidates whose mean reprojection error < reproj_threshold.
#           3. Single global merge pass across ALL candidates to eliminate duplicates
#              that arise from different person_data entries or camera pair overlaps.

#         Returns
#         -------
#         list of skeleton_3d  (np.ndarray, shape num_keypoints x 3)
#         """
#         all_candidates = []

#         for person_data in matched_persons:
#             person_match = person_data['ids']

#             new_3d_candidates, _ = self.generate_3d_candidates(all_annotations, person_match)

#             for candidate in new_3d_candidates:
#                 sk_objs     = self._create_skeleton_objects_for_person(candidate, all_annotations)
#                 valid_kps   = self._to_3d_keypoints_structure(sk_objs)
#                 skeleton_3d = self._reconstruct_points_from_svd(valid_kps)

#                 mean_error = self._compute_reprojection_error(skeleton_3d, candidate, all_annotations)

#                 if mean_error < reproj_threshold:
#                     non_zero = skeleton_3d[~np.all(skeleton_3d == 0, axis=1)]
#                     average_point = np.mean(non_zero, axis=0) if len(non_zero) > 0 else np.zeros(3)
#                     all_candidates.append({
#                         "skeleton_3d":   skeleton_3d,
#                         "matche_2d":     candidate,
#                         "average_point": average_point,
#                     })

#         # Single global merge eliminates duplicates across all person_data entries
#         return self._merge_duplicates(all_candidates, all_annotations)