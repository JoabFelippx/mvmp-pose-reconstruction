import numpy as np

class FundamentalMatrices:
    
    def __init__(self, camera_files, use_undistorted=True) -> None:
        self.camera_files = camera_files
        self.use_undistorted = use_undistorted
        
        if use_undistorted:
            print("FundamentalMatrices: Usando nK (imagens sem distorção)")
        else:
            print("FundamentalMatrices: Usando K (imagens com distorção)")
        
    def get_all_calibs_parameters(self):
        all_calibs_parameters = {}
        
        for c in range(len(self.camera_files)):
            rt, R, T, K, nK, dist_coeffs, h, w, roi = self._load_camera_parameters(self.camera_files[c])
            all_calibs_parameters[c] = {
                "rt": rt,
                "R": R,
                "T": T,
                "K": K,
                "nK": nK,
                "dist_coeffs": dist_coeffs,
                "roi": roi,
                "h": int(h),
                "w": int(w)
            }
            
        return all_calibs_parameters

    def _load_camera_parameters(self, calibration: str):
        camera_data = np.load(calibration)

        K = camera_data['K']
        rt = camera_data['rt']
        # rt[:, 3] *= 0.01 
        roi = camera_data['roi'] if 'roi' in camera_data else None
        h = 1080
        w = 1920
        R = rt[:3, :3]
        T = rt[:3, 3].reshape(3, 1)
        dist_coeffs = camera_data["dist"]
        

        # Usa nK se disponível, senão usa K
        if 'nK' in camera_data and self.use_undistorted:
            nK = camera_data['nK']
        else:
            nK = K
            if self.use_undistorted:
                print(f"AVISO: nK não encontrada em {calibration}. Usando K.") 
                
        return rt, R, T, K, nK, dist_coeffs, h, w, roi
        
    def _calculate_fundamental_matrix(self, K1: np.ndarray, K2: np.ndarray, 
                                     RT1: np.ndarray, RT2: np.ndarray):
        
        RT1_4x4 = np.vstack([RT1, [0, 0, 0, 1]])
        RT2_4x4 = np.vstack([RT2, [0, 0, 0, 1]])
        
        RT_2_1 = RT2_4x4 @ np.linalg.inv(RT1_4x4)
        
        R_rel = RT_2_1[0:3, 0:3]
        T_rel = RT_2_1[0:3, 3]
        
        skew_symm = np.array([
            [0, -T_rel[2], T_rel[1]], 
            [T_rel[2], 0, -T_rel[0]], 
            [-T_rel[1], T_rel[0], 0]
        ])
         
        essential_matrix = skew_symm @ R_rel

        F = (np.linalg.inv(K2).T) @ essential_matrix @ (np.linalg.inv(K1))
        return F
    
    def _calculate_projection_matrix(self, K: np.ndarray, T: np.ndarray):
        return K @ T
        
    def projection_matrices_all(self):
        P_all_cameras = dict()
        
        for c in range(len(self.camera_files)):
            rt, R, T, K, nK, dist_coeffs, h, w, roi = self._load_camera_parameters(self.camera_files[c])
            
            camera_matrix = nK if self.use_undistorted else K
            
            P = self._calculate_projection_matrix(camera_matrix, rt)
            P_all_cameras[c] = P
    
        return P_all_cameras
    
    def fundamental_matrices_all(self):
        F_all_camera_pairs = dict()
        
        for c_s in range(len(self.camera_files)):
            for c_d in range(len(self.camera_files)):
                if c_s == c_d:
                    continue
                
                rt1, R1, T1, K1, nK1, dist_coeffs, h1, w1, roi1 = self._load_camera_parameters(self.camera_files[c_s])
                rt2, R2, T2, K2, nK2, dist_coeffs, h2, w2, roi2 = self._load_camera_parameters(self.camera_files[c_d])

                cam_matrix_1 = nK1 if self.use_undistorted else K1
                cam_matrix_2 = nK2 if self.use_undistorted else K2

                F = self._calculate_fundamental_matrix(cam_matrix_1, cam_matrix_2, rt1, rt2)
                
                if c_s not in F_all_camera_pairs:
                    F_all_camera_pairs[c_s] = {}
                F_all_camera_pairs[c_s][c_d] = F
                
        return F_all_camera_pairs
   