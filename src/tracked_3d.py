import numpy as np
from scipy.optimize import linear_sum_assignment



class KalmanCV2D:
    def __init__(self, acceleration_variance):
        self.state = None
        self.covariance = None
        self.timestamp = None
        self.acceleration_variance = acceleration_variance

    @staticmethod
    def transition_matrix(dt):
        F = np.array([
                [1.0, 0.0, dt, 0.0 ],
                [0.0, 1.0, 0.0, dt ],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float64)
        return F

    def process_noise_matrix(self, dt):
        Q = self.acceleration_variance * np.array([
            [dt**4 / 4, 0, dt**3 / 2, 0],
            [0, dt**4 / 4, 0, dt**3 / 2],
            [dt**3 / 2, 0, dt**2, 0],
            [0, dt**3 / 2, 0, dt**2],
        ], dtype=np.float64)
        return Q

    def initialize(self, position_xy, timestamp, position_covariance, initial_velocity_variance):
        # Inicialmente
        self.state = np.array([
            [position_xy[0]],
            [position_xy[1]],
            [0],
            [0]
        ], dtype=np.float64)

        self.covariance = np.zeros((4,4), dtype=np.float64)
        self.covariance[:2, :2] = position_covariance
        self.covariance[2, 2] = initial_velocity_variance
        self.covariance[3, 3] = initial_velocity_variance

        self.timestamp = timestamp

    def predict(self, timestamp):

        if self.state is None:
            raise RuntimeError("Filtro ainda não inicializado")

        if self.covariance is None:
            raise RuntimeError("Covariância ainda não inicializada")

        if self.timestamp is None:
            raise RuntimeError("Timestamp ainda não inicializado")

        dt = timestamp - self.timestamp

        if dt < 0:
            raise ValueError(f"Timestamp fora de ordem: dt={dt}")

        if dt == 0:
            return 0.0

        transition_motion = self.transition_matrix(dt)
        uncertainty = self.process_noise_matrix(dt)

        self.state = transition_motion @ self.state
        self.covariance = transition_motion @ self.covariance @ transition_motion.T + uncertainty
        self.timestamp = timestamp


    def update(self, position_xy, measurement_covariance):

        if self.state is None:
            raise RuntimeError("Filtro ainda não inicializado")

        if self.covariance is None:
            raise RuntimeError("Covariância ainda não inicializada")

        if self.timestamp is None:
            raise RuntimeError("Timestamp ainda não inicializado")

        z = np.asarray(position_xy, dtype=np.float64).reshape(2, 1)

        H = np.zeros((2,4), dtype=np.float64)
        H[0, 0] = 1
        H[1, 1] = 1

        z_pred = H @ self.state

        innovation = z - z_pred

        S = H @ self.covariance @ H.T + measurement_covariance

        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation

        self.covariance = (np.identity(4) - K @ H) @ self.covariance

class SkeletonTracker3D:

    def __init__(self, acceleration_variance=0.25, measurement_covariance=None, initial_velocity_variance=1, max_distance=0.30, max_missed=10):
        self.tracks = {}
        self.next_id = 1
        self.max_distance = max_distance
        self.max_missed = max_missed
        
        self.acceleration_variance = acceleration_variance
        
        if measurement_covariance is None:
            self.measurement_covariance = np.array([[0.025**2, 0.0],[0.0, 0.025**2],], dtype=np.float64)
        else:
            self.measurement_covariance = np.asarray(measurement_covariance, dtype=np.float64)
            
        self.initial_velocity_variance = initial_velocity_variance
        
    def center_calculate(self, skeleton_3d):

        l_hip = skeleton_3d.get(12, None)
        r_hip = skeleton_3d.get(13, None)

        if l_hip is None or r_hip is None:
            return None

        l_hip = np.asarray(l_hip, dtype=np.float64)
        r_hip = np.asarray(r_hip, dtype=np.float64)

        center = (l_hip + r_hip) / 2.0
        return center[:2]

    def distance_calculate(self, track, detection):

        track_center = track["center"]
        detection_center = detection["center"]

        dist = np.linalg.norm(track_center - detection_center)

        return dist

    def build_detections(self, persons):

        detections = []

        for person in persons:

            skeleton_3d = person["skeleton_3d"]

            center = self.center_calculate(skeleton_3d)

            if center is None:
                continue

            detections.append({
                "person": person,
                "center": center,
            })

        return detections

    def create_track(self, detection, timestamp):

        track_id = self.next_id

        kalman = KalmanCV2D(
            self.acceleration_variance
        )
    
        center_xy = np.asarray(
            detection["center"],
            dtype=np.float64
        )

        kalman.initialize(
            position_xy=center_xy,
            timestamp=timestamp,
            position_covariance=self.measurement_covariance,
            initial_velocity_variance=self.initial_velocity_variance
        )

        self.tracks[track_id] = {
            "center": detection["center"],
            "kalman": kalman,
            "missed": 0
        }

        detection["person"]["id"] = track_id
        self.next_id += 1

    def initialize_tracks(self, detections, timestamp):

        for detection in detections:
            self.create_track(detection, timestamp)

    def predict_tracks(self, timestamp):

        for track in self.tracks.values():
    
            track["kalman"].predict(timestamp)
    
            track["center"] = (
                track["kalman"].state[:2, 0].copy()
            )
        

    def build_cost_matrix(self, detections):

        row = len(self.tracks)
        column = len(detections)

        cost_matrix = np.zeros(
            (row, column),
            dtype=np.float64
        )

        for idx_track, track_id in enumerate(self.tracks):

            track = self.tracks[track_id]

            for idx_detection, detection in enumerate(detections):

                distance = self.distance_calculate(
                    track,
                    detection
                )
                
                cost_matrix[idx_track, idx_detection] = distance
                
        return cost_matrix
        
    def match_tracks(self, detections, timestamp):

        cost_matrix = self.build_cost_matrix(detections)
        track_ids = list(self.tracks.keys())

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_tracks = set()
        matched_detections = set()

        for row_pos, col_pos in zip(row_ind, col_ind):

            track_id = track_ids[row_pos]
            detection = detections[col_pos]

            distance = cost_matrix[row_pos, col_pos]

            if distance > self.max_distance:
                continue

            matched_tracks.add(track_id)
            matched_detections.add(col_pos)

            track = self.tracks[track_id]

            track["kalman"].update(
                position_xy=detection["center"],
                measurement_covariance=self.measurement_covariance
            )
    
            track["center"] = (track["kalman"].state[:2, 0].copy())
            
            track["missed"] = 0
            
            detection["person"]["id"] = track_id

        for track_id in track_ids:
            if track_id not in matched_tracks:
                self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.max_missed:
                self.tracks.pop(track_id)

        for idx_detection in range(len(detections)):
            if idx_detection not in matched_detections:
                detection = detections[idx_detection]
                self.create_track(detection, timestamp)

    def all_missed(self):
    
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            self.tracks[track_id]["missed"] += 1
            if self.tracks[track_id]["missed"] > self.max_missed:
                self.tracks.pop(track_id)

    def update(self, persons, timestamp):

        detections = self.build_detections(persons)

        if not self.tracks:
            if detections:
                self.initialize_tracks(detections, timestamp)
            return persons

        self.predict_tracks(timestamp)

        if not detections:
            self.all_missed()
            return persons

        self.match_tracks(detections, timestamp)

        return persons