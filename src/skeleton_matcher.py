from typing import Dict
from collections import defaultdict

import numpy as np
import networkx as nx
from itertools import combinations


# =============================================================================
# Funções utilitárias vetorizadas
# =============================================================================

def compute_line_intersection_2d(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det
    return np.array([x, y])


def dist_p_l_vectorized(points, lines):
    numerators   = np.abs(lines[:, 0] * points[:, 0] + lines[:, 1] * points[:, 1] + lines[:, 2])
    denominators = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
    denominators[denominators == 0] = np.inf
    return numerators / denominators


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
    def __init__(self, fundamentals, matcher_params, num_cameras, num_keypoints):
        self.fundamentals   = fundamentals
        self.num_keypoints  = num_keypoints
        self.num_cameras    = num_cameras

        self.sigma_tolerance          = matcher_params['sigma_tolerance']
        self.max_error_per_joint      = matcher_params['max_error_per_joint']
        self.weight_quality           = matcher_params['weight_quality']
        self.weight_quantity          = matcher_params['weight_quantity']
        self.min_compatibility_score  = matcher_params['min_compatibility_score']
        self.top_k_cycle_candidates   = matcher_params['top_k_cycle_candidates']
        self.early_stop_cycle_score   = matcher_params['early_stop_cycle_score']
        self.weight_cycle             = matcher_params['weight_cycle']
        self.min_cycle_score          = matcher_params['min_cycle_score']
        self.max_intersection_dist    = matcher_params['max_intersection_dist']
        self.weight_distance          = matcher_params['weight_distance']
        self.weight_score             = matcher_params['weight_score']
        self.min_keypoints_for_grouping = matcher_params['min_keypoints_for_grouping']
        self.min_kp_ratio             = matcher_params['min_kp_ratio']

        self.kp_weights_arr = np.array([
            matcher_params['kp_weights'].get(i, 1.0) for i in range(num_keypoints)
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

    # -------------------------------------------------------------------------
    # Compatibilidade epipolar (sem alteração)
    # -------------------------------------------------------------------------
    def build_global_matrix(self, skeletons_by_cam, ids_by_cam):

        all_skeletons = []
        cam_offsets = {}
        offset = 0
        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            cam_offsets[cam_idx] = offset
            for sk_id in ids_by_cam[cam_idx]:
                all_skeletons.append((cam_idx, sk_id))
            offset += len(skeletons)
        m = len(all_skeletons)
        A_global_matrix = np.zeros((m, m))
        
        epilines_matrix = np.empty((m, m), dtype=object)
        
        return A_global_matrix, epilines_matrix, cam_offsets
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

    # -------------------------------------------------------------------------
    # Helper: epilinhas organizadas por keypoint a partir das matrizes globais
    # -------------------------------------------------------------------------

    def _organize_epilines_by_keypoint(self, cam_ref, idx_ref,
                                        scores_matrix, epilines_matrix, valid_mask,
                                        skeletons_by_cam):
        """
        Substitui organize_epilines_by_keypoint do dataclass.

        Retorna dict:
            { kp_idx : [(cam_other, idx_other, score, line_3), ...] }
        """
        lines_per_kp = {}

        for cam_other in range(self.num_cameras):
            if cam_other == cam_ref:
                continue
            for idx_other in range(len(skeletons_by_cam[cam_other])):
                if not valid_mask[cam_ref, idx_ref, cam_other, idx_other]:
                    continue

                score    = float(scores_matrix[cam_ref, idx_ref, cam_other, idx_other])
                epilines = epilines_matrix[cam_ref, idx_ref, cam_other, idx_other]  # (num_kp, 3)

                for kp_idx in range(self.num_keypoints):
                    line = epilines[kp_idx]
                    if not np.isnan(line).any():
                        lines_per_kp.setdefault(kp_idx, []).append(
                            (cam_other, idx_other, score, line)
                        )

        return lines_per_kp

    # -------------------------------------------------------------------------
    # simple_match  →  preenche as matrizes globais
    # -------------------------------------------------------------------------

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
    
    def index_to_cam_skeleton(self, index, cam_offsets, ids_by_cam):
        """
        Dado um índice da matriz global, retorna (cam_idx, skeleton_id)
        """
        sorted_cams = sorted(cam_offsets.items(), key=lambda x: x[1])
        
        cam_idx = None
        for i, (cam, offset) in enumerate(sorted_cams):
            next_offset = sorted_cams[i + 1][1] if i + 1 < len(sorted_cams) else float('inf')
            if offset <= index < next_offset:
                cam_idx = cam
                local_index = index - offset
                break
        
        skeleton_id = ids_by_cam[cam_idx][local_index]
        
        return cam_idx, skeleton_id
    
    def _calculate_epilines(self, sk_other, F_other_to_ref):
        """
        Calcula as linhas epipolares dos keypoints de sk_other projetadas na cam_ref.
        Retorna array (num_keypoints, 3) com [a, b, c] para cada keypoint.
        """
        valid_mask = ~np.all(sk_other == 0, axis=1)

        full_epilines = np.full((self.num_keypoints, 3), np.nan)

        if not np.any(valid_mask):
            return full_epilines

        pts_other = sk_other[valid_mask]
        ones = np.ones((pts_other.shape[0], 1))
        pts_other_h = np.hstack([pts_other, ones])

        epilines = (F_other_to_ref @ pts_other_h.T).T  # (n_valid, 3)

        full_epilines[valid_mask] = epilines

        return full_epilines
    
    def _intersect_epilines(self, epilines_list):

        num_lines = len(epilines_list)
        A = np.array([line[:2] for line in epilines_list])
        b = np.array([-line[2] for line in epilines_list])
        
        point, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        if num_lines >= 3:
            intersecoes_pares = []
            for i in range(num_lines):
                for j in range(i + 1, num_lines):
                    A_par = A[[i, j]]
                    b_par = b[[i, j]]
                    try:
                        p_par = np.linalg.solve(A_par, b_par)
                        intersecoes_pares.append(p_par)
                    except np.linalg.LinAlgError:
                        continue

            if len(intersecoes_pares) >= 2:
                distancias = []
                for k in range(len(intersecoes_pares)):
                    for l in range(k + 1, len(intersecoes_pares)):
                        distancias.append(np.linalg.norm(intersecoes_pares[k] - intersecoes_pares[l]))
                
                max_divergencia = max(distancias) if distancias else 0
                limite_concordancia = 20.0 
                
                if max_divergencia > limite_concordancia:
                    return np.array([1e6, 1e6])

        return point

    def refined_match(self, skeletons_by_cam, ids_by_cam, colum, line, cam_offsets, epilines_matrix):

        indices_por_coluna = {}
        for idx, col_val in enumerate(colum):
            if col_val not in indices_por_coluna:
                indices_por_coluna[col_val] = []
            indices_por_coluna[col_val].append(idx)

        for col_val, idxs in indices_por_coluna.items():

            cam_ref, sk_id_ref = self.index_to_cam_skeleton(col_val, cam_offsets, ids_by_cam)
            local_idx_ref = ids_by_cam[cam_ref].index(sk_id_ref)
            sk_ref = skeletons_by_cam[cam_ref][local_idx_ref]

            for i in idxs:
                row_idx = line[i]
                cam_other, sk_id_other = self.index_to_cam_skeleton(row_idx, cam_offsets, ids_by_cam)

                if cam_ref == cam_other:
                    continue

                local_idx_other = ids_by_cam[cam_other].index(sk_id_other)
                sk_other = skeletons_by_cam[cam_other][local_idx_other]

                F_other_to_ref = self.fundamentals[cam_other][cam_ref]
                epilines_on_ref = self._calculate_epilines(sk_other, F_other_to_ref)

                F_ref_to_other = self.fundamentals[cam_ref][cam_other]
                epilines_on_other = self._calculate_epilines(sk_ref, F_ref_to_other)

                epilines_matrix[row_idx, col_val] = epilines_on_ref
                epilines_matrix[col_val, row_idx] = epilines_on_other

        return epilines_matrix
    def intersection_affinity(self, skeletons_by_cam, ids_by_cam,
                               colum, line, cam_offsets, epilines_matrix, A_simple):
        """
        Retorna:
          A_intersection : np.ndarray (m, m)  – score de afinidade geométrica [0,1]
          A_detail       : dict  (i,j) → {'avg_dist', 'avg_cost', 'avg_score', 'count'}
                           guardado para o build_skeleton_groups usar métricas detalhadas
        """
        m              = epilines_matrix.shape[0]
        A_intersection = np.zeros((m, m))
        # Detalhe por par de índices globais para usar no build_skeleton_groups
        A_detail: Dict = {}

        # Mapeia col_val → lista de row_idx com epilinha calculada
        indices_por_coluna = {}
        for idx, col_val in enumerate(colum):
            indices_por_coluna.setdefault(col_val, []).append(idx)

        for col_val, idxs in indices_por_coluna.items():
            cam_ref, sk_id_ref = self.index_to_cam_skeleton(col_val, cam_offsets, ids_by_cam)
            local_ref          = ids_by_cam[cam_ref].index(sk_id_ref)
            sk_ref             = skeletons_by_cam[cam_ref][local_ref]

            # ── Passo 1: organizar epilinhas por keypoint ──────────────────────
            # Cada entrada: (cam_other, sk_id_other, simple_score, line_3)
            linhas_por_kp: Dict[int, list] = {}

            for i in idxs:
                row_idx = line[i]
                if row_idx == col_val:
                    continue

                cam_other, sk_id_other = self.index_to_cam_skeleton(row_idx, cam_offsets, ids_by_cam)
                if cam_other == cam_ref:
                    continue

                epilines = epilines_matrix[row_idx, col_val]
                if epilines is None:
                    continue

                simple_score = float(A_simple[row_idx, col_val])

                for kp_idx in range(self.num_keypoints):
                    line_kp = epilines[kp_idx]
                    if not np.isnan(line_kp).any():
                        linhas_por_kp.setdefault(kp_idx, []).append(
                            (cam_other, sk_id_other, simple_score, line_kp)
                        )

            # ── Passo 2-4: combinações par a par por keypoint ─────────────────
            # Acumula por par (cam_other, sk_id_other) quantos keypoints
            # apareceram como melhor combinação, mais custo / dist / score médios
            connections_count : Dict = defaultdict(int)
            connections_costs : Dict = defaultdict(float)
            connections_dists : Dict = defaultdict(float)
            connections_scores: Dict = defaultdict(float)

            valid_kp_count = int(np.sum(~np.all(sk_ref == 0, axis=1)))

            for kp_idx, data in linhas_por_kp.items():
                if len(data) < 2:
                    continue

                candidate_combinations = list(combinations(data, 2))
                graph_dists  = {}
                graph_scores = {}

                for (cam_1, skt_1, score_1, line_1), (cam_2, skt_2, score_2, line_2) in candidate_combinations:
                    point_intersect = compute_line_intersection_2d(line_1, line_2)
                    if point_intersect is None:
                        continue

                    # Distância do ponto a todas as linhas do keypoint
                    all_lines  = np.array([entry[3] for entry in data])
                    all_scores = np.array([entry[2] for entry in data])
                    dists_to_lines = dist_p_l_vectorized(
                        np.tile(point_intersect, (all_lines.shape[0], 1)),
                        all_lines
                    )
                    avg_dist  = float(np.mean(dists_to_lines))
                    avg_score = float(np.mean(all_scores))

                    if avg_dist > self.max_intersection_dist:
                        continue

                    key = ((cam_1, skt_1), (cam_2, skt_2))
                    graph_dists[key]  = avg_dist
                    graph_scores[key] = avg_score

                if not graph_scores:
                    continue

                best = self._select_best_combination(graph_scores, graph_dists)
                best['avg_dist'] = graph_dists[best['original_key']]

                for (cam_other, skt_other_id) in best['combination'].items():
                    k = (cam_other, skt_other_id)
                    connections_count[k]  += 1
                    connections_costs[k]  += best['combined_cost']
                    connections_dists[k]  += best['avg_dist']
                    connections_scores[k] += best['score']

            # ── Passo 5-6: threshold e preenchimento da A_intersection ─────────
            relative_threshold = max(
                self.min_keypoints_for_grouping,
                int(np.ceil(self.min_kp_ratio * valid_kp_count))
            )

            for (cam_other, skt_other_id), count in connections_count.items():
                if count < relative_threshold:
                    continue

                # Índice global do esqueleto other
                local_other = ids_by_cam[cam_other].index(skt_other_id)
                row_idx     = cam_offsets[cam_other] + local_other

                avg_cost  = connections_costs[(cam_other, skt_other_id)]  / count
                avg_dist  = connections_dists[(cam_other, skt_other_id)]  / count
                avg_score = connections_scores[(cam_other, skt_other_id)] / count

                # Score de interseção: penaliza custo (dist) e recompensa score epipolar
                intersection_score = 1.0 / (abs(avg_cost) + 1e-6)
                # Normaliza para [0,1] via sigmoid suave
                intersection_score = float(np.tanh(intersection_score / 10.0))

                # Preenche simétricamente
                A_intersection[row_idx, col_val] = intersection_score
                A_intersection[col_val, row_idx] = intersection_score

                pair = (min(row_idx, col_val), max(row_idx, col_val))
                A_detail[pair] = {
                    'avg_cost':  avg_cost,
                    'avg_dist':  avg_dist,
                    'avg_score': avg_score,
                    'count':     count,
                }

        return A_intersection, A_detail


    def build_skeleton_groups(self, A_combined, cam_offsets, skeletons_by_cam, ids_by_cam,
                               A_detail: Dict = None):
        G = nx.Graph()

        for cam_idx, skeletons in enumerate(skeletons_by_cam):
            for sk_id in ids_by_cam[cam_idx]:
                G.add_node((cam_idx, sk_id))

        rows, cols = np.nonzero(A_combined)
        for i, j in zip(rows, cols):
            if i >= j:
                continue

            score = A_combined[i, j]
            if score <= 0:
                continue

            cam_i, sk_id_i = self.index_to_cam_skeleton(i, cam_offsets, ids_by_cam)
            cam_j, sk_id_j = self.index_to_cam_skeleton(j, cam_offsets, ids_by_cam)

            if cam_i == cam_j:
                continue

            # Usa métricas detalhadas do A_detail quando disponíveis
            pair = (min(i, j), max(i, j))
            if A_detail and pair in A_detail:
                d        = A_detail[pair]
                avg_cost  = d['avg_cost']
                avg_dist  = d['avg_dist']
                avg_score = d['avg_score']
            else:
                avg_cost  = 1.0 - score
                avg_dist  = 0.0
                avg_score = score

            G.add_edge(
                (cam_i, sk_id_i), (cam_j, sk_id_j),
                weight=score,
                avg_cost=avg_cost,
                avg_dist=avg_dist,
                avg_score=avg_score,
            )

        matched_persons = []
        G_mst = nx.maximum_spanning_tree(G, weight='weight')

        for component in nx.connected_components(G_mst):
            nodes_by_camera = {}
            for raw_cam_id, skt_id in component:
                cam_id = int(raw_cam_id)
                nodes_by_camera.setdefault(cam_id, []).append((cam_id, skt_id))

            group_ids = {}
            for cam_id, nodes in nodes_by_camera.items():
                if len(nodes) == 1:
                    group_ids[cam_id] = int(nodes[0][1])
                else:
                    best_node = max(nodes, key=lambda n: G_mst.degree(n, weight='weight'))
                    group_ids[cam_id] = int(best_node[1])

            if len(group_ids) < 2:
                continue

            total_cost = total_dist = total_score = total_edges = 0
            subgraph = G_mst.subgraph(component)
            for n1, n2, data in subgraph.edges(data=True):
                total_cost  += data['avg_cost']
                total_dist  += data['avg_dist']
                total_score += data['avg_score']
                total_edges += 1

            matched_persons.append({
                'ids':         group_ids,
                'score':       total_score / total_edges if total_edges > 0 else 0,
                'avg_cost':    total_cost  / total_edges if total_edges > 0 else 0,
                'avg_dist':    total_dist  / total_edges if total_edges > 0 else 0,
                'num_cameras': len(group_ids),
                'num_edges':   total_edges,
            })

        matched_persons.sort(key=lambda x: (-x['num_cameras'], x['avg_cost']))
        return matched_persons

    def _validate_with_cycle_consistency(self, A_matrix, cam_offsets, skeletons_by_cam, ids_by_cam):

        row_nonzero, colum_nonzero = np.nonzero(A_matrix)[0], np.nonzero(A_matrix)[1]
        m = A_matrix.shape[0]

        score_cache = {}
        for row_val in row_nonzero:
            for col_val in colum_nonzero:
                key = tuple([(row_val, col_val), (col_val, row_val)])
                score_cache[key] = A_matrix[row_val, col_val]
                if row_val == col_val:
                    score_cache[key] = 0
        
        def get_compatibility_score(row_idx, col_idx):

            key = tuple([(row_idx, col_idx),(col_idx,row_idx)])
            if key in score_cache:
                return score_cache[key]
            
            cam_a, sk_id_a = self.index_to_cam_skeleton(row_idx, cam_offsets, ids_by_cam)
            cam_b, sk_id_b = self.index_to_cam_skeleton(col_idx, cam_offsets, ids_by_cam)

            sk_a = skeletons_by_cam[cam_a][sk_id_a]
            sk_b = skeletons_by_cam[cam_b][sk_id_b]

            F_a_to_b = self.fundamentals[cam_a][cam_b]
            F_b_to_a = self.fundamentals[cam_b][cam_a]

            score, _ = self._calculate_skeleton_compatibility_vectorized(sk_a, sk_b, F_a_to_b, F_b_to_a)
            score_cache[key] = score
            return score

        for ref in range(m):
            cam_ref, sk_id_ref = self.index_to_cam_skeleton(ref, cam_offsets, ids_by_cam)
            
            row_ref = A_matrix[ref, :].copy()
            row_ref[ref] = 0

            # Candidatos para ref: índices com score > 0 na linha
            candidate_indices = np.nonzero(row_ref)[0]

            for other in candidate_indices:
                cam_other, sk_id_other = self.index_to_cam_skeleton(other, cam_offsets, ids_by_cam)

                # Ignorar esqueletos da mesma câmera
                if cam_other == cam_ref:
                    continue

                original_score = A_matrix[ref, other]

                best_cycle_score = 0
                found_good_cycle = False

                # Buscar terceiros candidatos: esqueletos que combinam com ref
                third_candidates_scores = []
                for third in range(m):
                    if third == ref or third == other:
                        continue
                    cam_third, _ = self.index_to_cam_skeleton(third, cam_offsets, ids_by_cam)
                    if cam_third == cam_ref or cam_third == cam_other:
                        continue
                    score_ref_third = get_compatibility_score(ref, third)
                    third_candidates_scores.append((third, score_ref_third))

                third_candidates_scores.sort(key=lambda x: x[1], reverse=True)
                top_thirds = [idx for idx, _ in third_candidates_scores[:self.top_k_cycle_candidates]]

                for third in top_thirds:
                    score_ref_other  = get_compatibility_score(ref, other)
                    score_other_third = get_compatibility_score(other, third)
                    score_ref_third   = get_compatibility_score(ref, third)

                    if score_ref_other > 0 and score_other_third > 0 and score_ref_third > 0:
                        cycle_score = (score_ref_other * score_other_third * score_ref_third) ** (1/3)
                    else:
                        cycle_score = 0

                    best_cycle_score = max(best_cycle_score, cycle_score)

                    if best_cycle_score > self.early_stop_cycle_score:
                        found_good_cycle = True
                        break

                # Combinar score original com o cycle score
                combined_score = (1 - self.weight_cycle) * original_score + self.weight_cycle * best_cycle_score

                # Atualizar ou zerar na matriz conforme threshold
                if best_cycle_score >= self.min_cycle_score:
                    A_matrix[ref, other] = combined_score
                    A_matrix[other, ref] = combined_score  # manter simetria
                else:
                    A_matrix[ref, other] = 0
                    A_matrix[other, ref] = 0

        return A_matrix

    def _select_best_combination(self, graph_scores, graph_dists):

        scores = np.array(list(graph_scores.values()))
        dists = np.array([graph_dists[k] for k in graph_scores.keys()])
        keys = list(graph_scores.keys())

        score_range = scores.max() - scores.min()
        dist_range = dists.max() - dists.min()

        if score_range > 1e-8:
            score_norm = (scores - scores.min()) / score_range
        else:
            # Se todos os scores são iguais, normaliza para 0.5
            score_norm = np.full_like(scores, 0.5)  

        if dist_range > 1e-8:
            dist_norm = (dists - dists.min()) / dist_range
        else:
            # Se todas as distâncias são iguais, normaliza para 0.0 (melhor caso)
            dist_norm = np.zeros_like(dists)

        weight_distance = self.weight_distance
        weight_score = self.weight_score

        total = weight_distance + weight_score

        if total > 0:
            weight_distance /= total
            weight_score /= total

        # Custo combinado (menor é melhor)
        # +dist_norm para MINIMIZAR distância (valores pequenos = bom)
        # -score_norm para MAXIMIZAR score (valores grandes = bom)
        combined_cost = weight_distance * dist_norm - weight_score * score_norm

        best_idx = np.argmin(combined_cost)
        best_key = keys[best_idx]

        return {
            'combination': dict(best_key),
            'original_key': best_key,
            'score': graph_scores[best_key],
            'combined_cost': combined_cost[best_idx]
        }

            
    def match_skeletons(self, skeletons_by_cam, ids_by_cam, use_cycle_consistency=True):

        total_skeletons = sum(len(s) for s in skeletons_by_cam)
        if total_skeletons == 0:
            return []

        # 1. Matriz global e offsets
        A_matrix_global, epilines_matrix, cam_offsets = self.build_global_matrix(
            skeletons_by_cam, ids_by_cam
        )

        # 2. Afinidade geométrica (epipolar)
        A_simple = self.simple_match(
            skeletons_by_cam, ids_by_cam, A_matrix_global, cam_offsets
        )

        A_simple = self._validate_with_cycle_consistency( A_simple, cam_offsets, skeletons_by_cam, ids_by_cam)

        # (Opcional) afinidade por interseção de epilinhas
        line = np.nonzero(A_simple)[0]
        colum = np.nonzero(A_simple)[1]
        epilines_matrix = self.refined_match(
            skeletons_by_cam, ids_by_cam, colum, line, cam_offsets, epilines_matrix
        )
        A_intersection, A_detail = self.intersection_affinity(
            skeletons_by_cam, ids_by_cam, colum, line, cam_offsets, epilines_matrix, A_simple
        )
        A_combined = A_simple * A_intersection
        
        matched_persons = self.build_skeleton_groups(
            A_combined, cam_offsets, skeletons_by_cam, ids_by_cam, A_detail=A_detail
        )
        return matched_persons