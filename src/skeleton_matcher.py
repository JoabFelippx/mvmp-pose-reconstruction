import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment


def sampson_error_vectorized(pts1_h, pts2_h, F):
    line_on_image_2 = (F @ pts1_h.T).T
    line_on_image_1 = (F.T @ pts2_h.T).T
    e = np.sum(pts2_h * (F @ pts1_h.T).T, axis=1)
    numerator = e ** 2
    denominator = (
        line_on_image_2[:, 0] ** 2 +
        line_on_image_2[:, 1] ** 2 +
        line_on_image_1[:, 0] ** 2 +
        line_on_image_1[:, 1] ** 2
    )
    denominator[denominator < 1e-12] = np.inf
    return numerator / denominator



class SkeletonMatcher:

    def __init__(self, fundamentals, matcher_params, num_cameras, num_keypoints,
                 camera_calibrations=None):
        self.fundamentals = fundamentals
        self.num_keypoints = num_keypoints
        self.num_cameras = num_cameras

        # Parâmetros realmente usados pelo pipeline atual.
        self.sigma_tolerance = matcher_params["sigma_tolerance"]
        self.max_error_per_joint = matcher_params["max_error_per_joint"]
        self.weight_quality = matcher_params["weight_quality"]
        self.weight_quantity = matcher_params["weight_quantity"]
        self.min_compatibility_score = matcher_params["min_compatibility_score"]
        self.weight_cycle = matcher_params["weight_cycle"]
        self.weight_distance = matcher_params.get("weight_distance", 0.60)
        self.weight_score = matcher_params.get("weight_score", 0.40)
        self.distance_d0 = matcher_params.get("distance_d0", 1.25)

        # As calibrações usam os índices locais das câmeras, como fundamentals.
        # Guarde as transformações inversas para projetar os pés no plano z=0.
        self.ground_projection = {}
        for cam_idx, calibration in (camera_calibrations or {}).items():
            intrinsic = np.asarray(calibration["nK"], dtype=np.float64)
            rt = np.asarray(calibration["rt"], dtype=np.float64)
            rt4 = np.vstack((rt, [0.0, 0.0, 0.0, 1.0]))
            world_from_camera = np.linalg.inv(rt4)
            self.ground_projection[cam_idx] = (
                np.linalg.inv(intrinsic),
                world_from_camera[:3, :3],
                world_from_camera[:3, 3],
            )

        self.kp_weights_arr = np.array([
            matcher_params["kp_weights"].get(i, 1.0)
            for i in range(num_keypoints)
        ])


    def extract_skeletons_from_annotations(self, annotations):
        skeletons_by_cam = []
        ids_by_cam       = []

        for i in range(self.num_cameras):
            skeletons_for_current_cam = []
            ids_for_current_cam       = []

            if i < len(annotations):
                for obj in annotations[i].objects:
                    skeleton = np.zeros((self.num_keypoints, 2), dtype=np.float32)
                    for kp in obj.keypoints:
                        if kp.id < self.num_keypoints:
                            skeleton[kp.id - 1] = [kp.position.x, kp.position.y]
                    skeletons_for_current_cam.append(skeleton)
                    ids_for_current_cam.append(obj.id)

            skeletons_by_cam.append(skeletons_for_current_cam)
            ids_by_cam.append(ids_for_current_cam)

        return skeletons_by_cam, ids_by_cam

    def build_global_matrix(self, skeletons_by_cam):
        """Cria a matriz global de afinidade e os offsets de cada câmera."""
        cam_offsets = {}
        offset = 0

        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            cam_offsets[cam_idx] = offset
            offset += len(skeletons)

        affinity_matrix = np.zeros((offset, offset), dtype=np.float64)
        return affinity_matrix, cam_offsets

    def _ground_center(self, skeleton, cam_idx):
        """Projeta no chão o ponto sob o quadril e os tornozelos detectados."""
        if cam_idx not in self.ground_projection:
            return None

        valid = np.all(np.isfinite(skeleton), axis=1) & np.any(skeleton != 0, axis=1)
        if not np.any(valid):
            return None

        # IDs COCO 1-based: quadris 12/13, tornozelos 16/17.
        hips = [idx for idx in (11, 12) if idx < len(skeleton) and valid[idx]]
        ankles = [idx for idx in (15, 16) if idx < len(skeleton) and valid[idx]]
        u = np.mean(skeleton[hips, 0]) if hips else np.mean(skeleton[valid, 0])
        v = np.max(skeleton[ankles, 1]) if ankles else np.max(skeleton[valid, 1])

        intrinsic_inv, rotation, camera_center = self.ground_projection[cam_idx]
        ray = rotation @ (intrinsic_inv @ np.array([u, v, 1.0]))
        if abs(ray[2]) < 1e-12:
            return None
        center = camera_center[:2] - camera_center[2] * ray[:2] / ray[2]
        return center if np.all(np.isfinite(center)) else None

    def distance_affinity(self, skeletons_by_cam, cam_offsets):
        """Afinidade sigmoide da distância no plano do chão (metros)."""
        total = sum(len(skeletons) for skeletons in skeletons_by_cam)
        affinity = np.full((total, total), np.nan, dtype=np.float64)
        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            start = cam_offsets[cam_idx]
            affinity[start:start + len(skeletons), start:start + len(skeletons)] = 0.0
        np.fill_diagonal(affinity, 1.0)

        centers = {}
        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            for idx, skeleton in enumerate(skeletons):
                center = self._ground_center(np.asarray(skeleton), cam_idx)
                if center is not None:
                    centers[cam_offsets[cam_idx] + idx] = center

        for cam_i, cam_j in combinations(range(self.num_cameras), 2):
            for idx_i in range(len(skeletons_by_cam[cam_i])):
                i = cam_offsets[cam_i] + idx_i
                if i not in centers:
                    continue
                for idx_j in range(len(skeletons_by_cam[cam_j])):
                    j = cam_offsets[cam_j] + idx_j
                    if j not in centers:
                        continue
                    distance = np.linalg.norm(centers[i] - centers[j])
                    x = np.clip(20.0 * (distance - self.distance_d0), -60.0, 60.0)
                    affinity[i, j] = affinity[j, i] = 1.0 / (1.0 + np.exp(x))

        return affinity


    def _calculate_skeleton_compatibility_vectorized(self, sk1, sk2, F_1_to_2, F_2_to_1):
        valid_mask = ~((np.all(sk1 == 0, axis=1)) | (np.all(sk2 == 0, axis=1)))

        if not np.any(valid_mask):
            return 0.0, None

        pts1    = sk1[valid_mask]
        pts2    = sk2[valid_mask]
        weights = self.kp_weights_arr[valid_mask]

        ones    = np.ones((pts1.shape[0], 1))
        pts1_h  = np.hstack([pts1, ones])
        pts2_h  = np.hstack([pts2, ones])

        lines_on_1     = (F_2_to_1 @ pts2_h.T).T
        sampson_errors = sampson_error_vectorized(pts1_h, pts2_h, F_1_to_2)
        sampson_abs    = np.sqrt(sampson_errors)

        mae           = np.mean(sampson_abs)
        quality_score = np.exp(-mae / self.sigma_tolerance)

        valid_joints_mask = sampson_abs < self.max_error_per_joint
        num_valid_joints  = np.sum(weights[valid_joints_mask])
        max_possible      = np.sum(weights)
        quantity_score    = num_valid_joints / max_possible if max_possible > 0 else 0

        combined_score    = self.weight_quality * quality_score + self.weight_quantity * quantity_score

        full_lines_on_1 = np.full((self.num_keypoints, 3), np.nan)
        full_lines_on_1[valid_mask] = lines_on_1

        return combined_score, full_lines_on_1

    def simple_match(self, skeletons_by_cam, ids_by_cam, global_matrix, cam_offsets):

        affinity_matrix = global_matrix
        cam_offsets = cam_offsets
        
        for cam_ref, F_ref_to_others in self.fundamentals.items():
            
            ids_ref_cam = ids_by_cam[cam_ref]
            skeletons_ref_cam = skeletons_by_cam[cam_ref]

            for idx_skt_ref, skt_ref in enumerate(skeletons_ref_cam):
                skt_ref_id = ids_ref_cam[idx_skt_ref]
                affinity_matrix[cam_offsets[cam_ref] + idx_skt_ref, cam_offsets[cam_ref] + idx_skt_ref] = 1
                for cam_other, F_ref_to_other in F_ref_to_others.items():
                
                    skeletons_other_cam = skeletons_by_cam[cam_other]
                    ids_other_cam = ids_by_cam[cam_other]

                    F_other_to_ref = self.fundamentals[cam_other][cam_ref]
                    for idx_skt_other, skt_other in enumerate(skeletons_other_cam):
                        skt_other_id = ids_other_cam[idx_skt_other]
                        compatibility_score, all_epiline_on_ref = self._calculate_skeleton_compatibility_vectorized(skt_ref, skt_other, F_ref_to_other, F_other_to_ref)
                        if compatibility_score >= self.min_compatibility_score:

                            def update_affinity(i, j, score):
                                """Atualiza a matriz de afinidade de forma simétrica."""
                                if affinity_matrix[i, j] == 0:
                                    affinity_matrix[i, j] = score
                                else:
                                    affinity_matrix[i, j] = (affinity_matrix[i, j] + score) / 2

                            i_ref = cam_offsets[cam_ref] + idx_skt_ref
                            i_other = cam_offsets[cam_other] + idx_skt_other

                            update_affinity(i_ref, i_other, compatibility_score)
                            update_affinity(i_other, i_ref, compatibility_score)

        return affinity_matrix

    def _validate_with_cycle_consistency(self, A_matrix, cam_offsets, skeletons_by_cam, ids_by_cam):

        A_original = A_matrix.copy()
        A_new = A_matrix.copy()

        num_cameras = self.num_cameras

        for cam_i, cam_j in combinations(range(num_cameras), 2):

            n_i = len(skeletons_by_cam[cam_i])
            n_j = len(skeletons_by_cam[cam_j])

            if n_i == 0 or n_j == 0:
                continue

            start_i = cam_offsets[cam_i]
            start_j = cam_offsets[cam_j]
            
            # Bloco A_ij
            A_ij = A_original[
                start_i:start_i + n_i,
                start_j:start_j + n_j
            ]

            # Acumular suporte fornecido
            # pelas outras câmeras
            supports = []

            for cam_k in range(num_cameras):

                if cam_k == cam_i or cam_k == cam_j:
                    continue

                n_k = len(skeletons_by_cam[cam_k])

                if n_k == 0:
                    continue

                start_k = cam_offsets[cam_k]

                A_ik = A_original[
                    start_i:start_i + n_i,
                    start_k:start_k + n_k
                ]

                A_kj = A_original[
                    start_k:start_k + n_k,
                    start_j:start_j + n_j
                ]

                products = (A_ik[:, :, None] * A_kj[None, :, :])

                # melhor caminho i -> k -> j
                support_k = np.max(products, axis=1)

                supports.append(support_k)
            
            if not supports:
                continue

            supports = np.stack(supports, axis=0)

            cycle_support = np.max(supports, axis=0)

            alpha = self.weight_cycle

            combined = (1.0 - alpha) * A_ij + alpha * cycle_support

            valid = A_ij > 0

            block = np.zeros_like(A_ij)

            block[valid] = combined[valid]
            weak_direct = A_ij < 0.65
            weak_cycle = cycle_support < 0.40

            block[valid & weak_direct & weak_cycle ] = 0

            A_new[
                start_i:start_i + n_i,
                start_j:start_j + n_j
            ] = block

            A_new[
                start_j:start_j + n_j,
                start_i:start_i + n_i
            ] = block.T

        return A_new

    def hungarian_pairwise(self, A_matrix, cam_a, cam_b, cam_offsets, skeletons_by_cam, ids_by_cam, min_score=0.30):

        n_a = len(skeletons_by_cam[cam_a])
        n_b = len(skeletons_by_cam[cam_b])

        if n_a == 0 or n_b == 0:
            return []

        start_a = cam_offsets[cam_a]
        start_b = cam_offsets[cam_b]

        affinity = A_matrix[
            start_a:start_a + n_a,
            start_b:start_b + n_b
        ]

        if affinity.size == 0:
            return []
        
        cost = 1.0 - affinity

        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []

        for idx_a, idx_b in zip(row_ind, col_ind):

            score = float(affinity[idx_a, idx_b])

            if score < min_score: continue

            matches.append({
                "cam_a": cam_a,
                "cam_b": cam_b,

                "idx_a": int(idx_a),
                "idx_b": int(idx_b),

                "id_a": ids_by_cam[cam_a][idx_a],
                "id_b": ids_by_cam[cam_b][idx_b],

                "score":score,})
        return matches

    def hungarian_all_pairs(self, A_matrix, cam_offsets, skeletons_by_cam, ids_by_cam, min_score=0.3):

        all_matches = []

        for cam_a, cam_b in combinations(range(self.num_cameras), 2):

            matches = self.hungarian_pairwise(
                A_matrix=A_matrix,
                cam_a=cam_a,
                cam_b=cam_b,
                cam_offsets=cam_offsets,
                skeletons_by_cam=skeletons_by_cam,
                ids_by_cam=ids_by_cam,
                min_score=min_score,
            )

            all_matches.extend(matches)
        
        all_matches.sort(
            key=lambda m: m["score"],
            reverse=True
        )

        return all_matches

    def build_groups_from_hungarian_matches(self, pairwise_matches,):

        parent = {}
        group_cameras = {}

        def make(node):

            if node not in parent:
                parent[node] = node
                group_cameras[node] = {node[0]}

        def find(node):

            if parent[node] != node:
                parent[node] = find(parent[node])

            return parent[node]

        def union(node_a, node_b):

            make(node_a)
            make(node_b)

            root_a = find(node_a)
            root_b = find(node_b)

            if root_a == root_b:
                return True

            cameras_a = group_cameras[root_a]
            cameras_b = group_cameras[root_b]

            # Se existe câmera repetida,
            # unir esses grupos criaria conflito.
            if not cameras_a.isdisjoint(cameras_b):
                return False

            parent[root_b] = root_a

            group_cameras[root_a] = (cameras_a | cameras_b)

            del group_cameras[root_b]

            return True

        # Os matches já chegam ordenados
        # do melhor score para o pior.
        for match in pairwise_matches:

            node_a = (match["cam_a"], match["id_a"])
            node_b = (match["cam_b"], match["id_b"])

            union(node_a, node_b)

        # ----------------------------------
        # recuperar componentes
        # ----------------------------------

        components = {}

        for node in parent:

            root = find(node)

            components.setdefault(root, []).append(node)

        matched_persons = []

        for nodes in components.values():

            # triangulação exige pelo menos
            # duas câmeras
            if len(nodes) < 2:
                continue

            ids = {}

            for cam_idx, skeleton_id in nodes:

                ids[cam_idx] = skeleton_id

            matched_persons.append({
                "ids": ids
            })

        return matched_persons



    def match_skeletons(
        self,
        skeletons_by_cam,
        ids_by_cam,
        use_cycle_consistency=True,
    ):
        """
        Pipeline atual:

        afinidade epipolar -> suporte de ciclo (opcional) -> distância no chão -> Hungarian pairwise ->
        agrupamento global com restrição
        de no máximo uma detecção por câmera.
        """
        total_skeletons = sum(len(s) for s in skeletons_by_cam)
        if total_skeletons == 0:
            return []

        # 1. Matriz global e offsets.
        A_matrix_global, cam_offsets = self.build_global_matrix(
            skeletons_by_cam
        )

        # 2. Afinidade geométrica epipolar.
        A_simple = self.simple_match(
            skeletons_by_cam,
            ids_by_cam,
            A_matrix_global,
            cam_offsets,
        )

        # 3. Refinamento por suporte de ciclo, quando habilitado.
        if use_cycle_consistency:
            A_simple = self._validate_with_cycle_consistency(
                A_simple, cam_offsets, skeletons_by_cam, ids_by_cam
            )

        # 4. Combine as afinidades onde a projeção no chão está disponível.
        # Sem projeção válida, mantenha o score epipolar para evitar NaN no Hungarian.
        A_distance = self.distance_affinity(skeletons_by_cam, cam_offsets)
        A_final = A_simple.copy()
        valid_distance = np.isfinite(A_distance)
        A_final[valid_distance] = np.sqrt(
            A_distance[valid_distance] * A_simple[valid_distance]
        )

        # 5. Matching pairwise com Hungarian.
        pairwise_matches = self.hungarian_all_pairs(
            A_matrix=A_final,
            cam_offsets=cam_offsets,
            skeletons_by_cam=skeletons_by_cam,
            ids_by_cam=ids_by_cam,
            min_score=self.min_compatibility_score,
        )

        # 6. Agrupamento global respeitando uma detecção por câmera.
        return self.build_groups_from_hungarian_matches(
            pairwise_matches
        )
