import cv2
import numpy as np

def get_skeleton_center(skeleton_3d: dict, hip_indices: list = [11, 12]):
    
    hip_points = [skeleton_3d.get(idx) for idx in hip_indices if skeleton_3d.get(idx) is not None]
    
    if len(hip_points) < 1:
        return None
    
    center_point = np.mean(np.array(hip_points), axis=0)
    return center_point

def calib_img_from_file(npzCalib, image):
    """Aplica a calibração para remover a distorção da imagem."""
    undistort_img = cv2.undistort(image, npzCalib['K'], npzCalib['dist'], None, npzCalib['nK'])
    if 'roi' in npzCalib:
        x, y, w, h = npzCalib['roi']
        return undistort_img[y:y+h, x:x+w]
    return undistort_img
    
def draw_bounding_box(image, bbox_xyxy, color=(150, 0, 0), thickness=2):
    return cv2.rectangle(image, (int(bbox_xyxy[0]), int(bbox_xyxy[1])), (int(bbox_xyxy[2]), int(bbox_xyxy[3])), color, thickness)

def draw_identifier(image, bbox_xyxy, person_id, color=(150, 0, 0)):

    offset_x = 0
    offset_y = -10
    font_size = 1
    font_thickness = 2
    
    position = (int(bbox_xyxy[0] + offset_x), int(bbox_xyxy[1] + offset_y))
    
    return cv2.putText(image, str(person_id), position, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, font_thickness)

def draw_skeleton(image, keypoints, skeleton_map):
    for connection in skeleton_map:
        srt_kpt_id = connection['srt_kpt_id']
        dst_kpt_id = connection['dst_kpt_id']
        
        # Pega os pontos de início e fim
        p1 = keypoints[srt_kpt_id]
        p2 = keypoints[dst_kpt_id]

        # Verifica se ambos os pontos foram detectados
        if (p1[0] == 0 and p1[1] == 0) or (p2[0] == 0 and p2[1] == 0):
            continue

        color = connection.get('color', (0, 255, 0))
        thickness = connection.get('thickness', 2)

        cv2.line(image, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    return image

def draw_keypoints(image, keypoints, kpt_color_map):
    for kpt_id, data in kpt_color_map.items():
        point = keypoints[kpt_id]
        
        # Verifica se o ponto foi detectado
        if point[0] == 0 and point[1] == 0:
            continue

        color = data.get('color', (0, 0, 255))
        radius = data.get('radius', 4)

        cv2.circle(image, (int(point[0]), int(point[1])), radius, color, -1)
    return image

def create_adaptive_camera_grid(
    frames: list,
    cell_height: int = 250,
    cell_width: int = 400,
    add_labels: bool = True,
    label_font_scale: float = 0.8,
    label_color: tuple = (0, 255, 0),
    label_thickness: int = 2
) -> np.ndarray:
    """
    Cria um grid adaptativo de câmeras baseado no número de frames.
    
    Layouts automáticos:
    - 1 câmera:  1×1
    - 2 câmeras: 1×2 (horizontal)
    - 3 câmeras: 1×3 (horizontal)
    - 4 câmeras: 2×2
    - 5 câmeras: 2×3 (3 em cima, 2 embaixo + blank)
    - 6 câmeras: 2×3
    - 7 câmeras: 2×4 (4 em cima, 3 embaixo + blank)
    - 8 câmeras: 2×4
    - 9+ câmeras: até 3 linhas × ceil(n/3) colunas
    
    Args:
        frames: Lista de frames (numpy arrays BGR)
        cell_height: Altura desejada de cada célula
        cell_width: Largura desejada de cada célula
        add_labels: Se True, adiciona "Cam 1", "Cam 2", etc
        label_font_scale: Tamanho da fonte dos labels
        label_color: Cor RGB dos labels
        label_thickness: Espessura da fonte
        
    Returns:
        Grid montado como numpy array BGR
    """
    
    if not frames:
        # Retorna imagem preta pequena
        return np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    
    num_cams = len(frames)
    
    # ── 1. Resize todas as câmeras ────────────────────────────────────
    imgs_resized = []
    for i, frame in enumerate(frames):
        img_resized = cv2.resize(
            frame, 
            (cell_width, cell_height), 
            interpolation=cv2.INTER_NEAREST
        )
        
        if add_labels:
            cv2.putText(
                img_resized, 
                f"Cam {i+1}", 
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 
                label_font_scale, 
                label_color, 
                label_thickness
            )
        
        imgs_resized.append(img_resized)
    
    # ── 2. Criar blank para preencher espaços vazios ──────────────────
    blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
    
    # ── 3. Montar grid baseado no número de câmeras ───────────────────
    
    if num_cams == 1:
        return imgs_resized[0]
    
    elif num_cams == 2:
        return np.hstack(imgs_resized)
    
    elif num_cams == 3:
        return np.hstack(imgs_resized)
    
    elif num_cams == 4:
        linha1 = np.hstack([imgs_resized[0], imgs_resized[1]])
        linha2 = np.hstack([imgs_resized[2], imgs_resized[3]])
        return np.vstack([linha1, linha2])
    
    elif num_cams == 5:
        linha1 = np.hstack([imgs_resized[0], imgs_resized[1], imgs_resized[2]])
        linha2 = np.hstack([imgs_resized[3], imgs_resized[4], blank])
        return np.vstack([linha1, linha2])
    
    elif num_cams == 6:
        linha1 = np.hstack([imgs_resized[0], imgs_resized[1], imgs_resized[2]])
        linha2 = np.hstack([imgs_resized[3], imgs_resized[4], imgs_resized[5]])
        return np.vstack([linha1, linha2])
    
    elif num_cams == 7:
        linha1 = np.hstack([imgs_resized[0], imgs_resized[1], imgs_resized[2], imgs_resized[3]])
        linha2 = np.hstack([imgs_resized[4], imgs_resized[5], imgs_resized[6], blank])
        return np.vstack([linha1, linha2])
    
    elif num_cams == 8:
        linha1 = np.hstack(imgs_resized[0:4])
        linha2 = np.hstack(imgs_resized[4:8])
        return np.vstack([linha1, linha2])
    
    else:
        # 9+ câmeras: grid genérico
        # Estratégia: até 3 linhas, dividindo igualmente
        n_cols = math.ceil(num_cams / 3)
        rows = []
        
        for row_idx in range(3):
            start_idx = row_idx * n_cols
            end_idx = min(start_idx + n_cols, num_cams)
            
            if start_idx >= num_cams:
                break
            
            row_cams = imgs_resized[start_idx:end_idx].copy()
            
            # Preencher linha com blanks
            blanks_needed = n_cols - len(row_cams)
            for _ in range(blanks_needed):
                row_cams.append(blank.copy())
            
            rows.append(np.hstack(row_cams))
        
        return np.vstack(rows)