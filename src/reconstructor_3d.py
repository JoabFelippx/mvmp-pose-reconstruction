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

        return person_data
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
        
    def compute_reprojection_metrics(self, skeleton_3d, person_match, all_annotations):
        all_errors = []
        errors_by_camera = {}
        errors_by_joint = {}

        for cam_idx, skeleton_id in person_match.items():

            if cam_idx >= len(all_annotations):
                continue

            # Encontra o esqueleto 2D usado nessa câmera
            skeleton_2d = None

            for obj in all_annotations[cam_idx].objects:
                if obj.id == skeleton_id:
                    skeleton_2d = obj
                    break

            if skeleton_2d is None:
                continue

            P = self.projection_matrices[cam_idx]

            camera_errors = []

            for kp in skeleton_2d.keypoints:

                kp_id = kp.id

                # Esse joint precisa existir no esqueleto reconstruído
                if kp_id not in skeleton_3d:
                    continue

                observed = np.array(
                    [kp.position.x, kp.position.y],
                    dtype=np.float64
                )

                if np.allclose(observed, [0, 0]):
                    continue

                Xw = np.asarray(
                    skeleton_3d[kp_id],
                    dtype=np.float64
                )

                # [X Y Z 1]
                Xw_h = np.array(
                    [Xw[0], Xw[1], Xw[2], 1.0],
                    dtype=np.float64
                )

                # Reprojeta 3D -> 2D
                x_h = P @ Xw_h

                if abs(x_h[2]) < 1e-12:
                    continue

                projected = x_h[:2] / x_h[2]

                # Distância euclidiana em pixels
                error = float(
                    np.linalg.norm(projected - observed)
                )

                all_errors.append(error)
                camera_errors.append(error)

                errors_by_joint.setdefault(
                    kp_id, []
                ).append(error)

            if camera_errors:
                errors_by_camera[cam_idx] = {
                    "mean_px": float(
                        np.mean(camera_errors)
                    ),
                    "median_px": float(
                        np.median(camera_errors)
                    ),
                    "rmse_px": float(
                        np.sqrt(
                            np.mean(
                                np.square(camera_errors)
                            )
                        )
                    ),
                    "num_points": len(camera_errors),
                }

        if not all_errors:
            return {
                "mean_px": float("nan"),
                "median_px": float("nan"),
                "rmse_px": float("nan"),
                "p95_px": float("nan"),
                "num_points": 0,
                "per_camera": {},
                "per_joint": {},
            }

        errors = np.asarray(all_errors)

        per_joint = {}

        for kp_id, values in errors_by_joint.items():

            values = np.asarray(values)

            per_joint[kp_id] = {
                "mean_px": float(
                    np.mean(values)
                ),
                "rmse_px": float(
                    np.sqrt(
                        np.mean(values ** 2)
                    )
                ),
                "num_observations": int(
                    len(values)
                ),
            }

        return {
            "mean_px": float(np.mean(errors)),
            "median_px": float(np.median(errors)),
            "rmse_px": float(
                np.sqrt(np.mean(errors ** 2))
            ),
            "p95_px": float(
                np.percentile(errors, 95)
            ),
            "num_points": int(len(errors)),
            "per_camera": errors_by_camera,
            "per_joint": per_joint,
        }


    def reconstruct_all(self, matched_persons, all_annotations):
        
        
        reconstructed_skeletons = []

        for person_data in matched_persons:
            person_match = person_data['ids']

            skeleton_objects = self._create_skeleton_objects_for_person(person_match, all_annotations)
            valid_kps = self._to_3d_keypoints_structure(skeleton_objects)
            person_3d = self._reconstruct_points_from_svd(valid_kps)

            if not person_3d:
                continue

            reprojection = self.compute_reprojection_metrics(
            person_3d,
            person_match,
            all_annotations
            )

            reconstructed_skeletons.append({
                "skeleton_3d": person_3d,
                "matche_2d": person_match,
                "reprojection": reprojection,
            })

        return reconstructed_skeletons
