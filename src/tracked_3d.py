import numpy as np  
from scipy.optimize import linear_sum_assignment

class SkeletonTracker3D:

    def __init__(self, max_distance, max_missed=10):
        self.tracks = {}
        self.next_id = 1

        self.max_distance = max_distance
        self.max_missed = max_missed

    def center_calculate(self, person_3d):
        l_hip = person_3d.get(12, None)
        r_hip = person_3d.get(13, None)

        if l_hip is None or r_hip is None:
            return None

        l_hip = np.asarray(l_hip, dtype=np.float64)
        r_hip = np.asarray(r_hip, dtype=np.float64)

        center = (l_hip + r_hip) / 2.0
        return center

    def distance_calculate(self, tracked, detection_current):
        return np.linalg.norm(
            tracked["center"] - detection_current["center"]
        )
    def create_matrix(self, current_persons):
        row = len(self.tracks)
        col = len(current_persons)
        
        return np.zeros((row, col), dtype=np.float64)
    
    def update(self, persons):

        current_persons = []

        for person in persons:

            center = self.center_calculate(person["skeleton_3d"])

            if center is None:
                continue

            current_persons.append({
                "person": person,
                "center": center
            })

        if not self.tracks:

            for detection in current_persons:

                track_id = self.next_id


                self.tracks[self.next_id] = {
                    "center": detection["center"],
                    "missed": 0,
                }

                detection["person"]["id"] = self.next_id

                self.next_id += 1

            return persons

        A_dist = self.create_matrix(current_persons)

        for idx_tracked, tracked_person_id in enumerate(self.tracks):

            tracked = self.tracks[tracked_person_id]

            for idx_current, current_person in enumerate(current_persons):

                dist = self.distance_calculate(tracked, current_person)
                A_dist[idx_tracked, idx_current] = dist

        row_ind, col_ind = linear_sum_assignment(A_dist)

        track_ids = list(self.tracks.keys())

        matched_tracks = set()
        matched_detections = set()

        for idx_tracked, idx_current in zip(row_ind, col_ind):
            track_id = track_ids[idx_tracked]

            distance = A_dist[
                idx_tracked,
                idx_current
            ]

            if distance > self.max_distance:
                continue

            current_person = current_persons[idx_current]

            self.tracks[track_id]["center"] = current_person["center"]

            self.tracks[track_id]["missed"] = 0

            current_person["person"]["id"] = track_id

            matched_tracks.add(track_id)
            matched_detections.add(idx_current)

        for track_id in track_ids:

            if track_id not in matched_tracks:

                self.tracks[track_id]["missed"] += 1

        tracks_to_remove = []

        for track_id in track_ids:

            if track_id not in matched_tracks:

                self.tracks[track_id]["missed"] += 1

                if (
                    self.tracks[track_id]["missed"]
                    > self.max_missed
                ):
                    tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        for idx_current, current_person in enumerate(current_persons):

            if idx_current in matched_detections:
                continue

            track_id = self.next_id

            self.tracks[track_id] = {
                "center": current_person["center"],
                "missed": 0,
            }

            current_person["person"]["id"] = track_id

            self.next_id += 1

        return persons
