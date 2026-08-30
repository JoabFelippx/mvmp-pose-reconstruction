import argparse
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

def load_calibration(path):
    data = np.load(path)

    return {
        "K": data["K"],
        "nK": data["nK"],
        "dist": data["dist"],
        "roi": data["roi"] if "roi" in data else None,
    }

def undistort_frame(frame, calib):
    h, w = frame.shape[:2]

    map1, map2 = cv2.initUndistortRectifyMap(calib["K"], calib["dist"], None, calib["nK"], (w, h), cv2.CV_16SC2,)

    frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR,)

    roi = calib["roi"]

    if roi is not None:
        x, y, w_roi, h_roi = roi
        frame = frame[0:h_roi, 0:w_roi]

    return frame

def extract_persons(result):

    persons = []

    if result.keypoints is None:
        return persons

    xy = result.keypoints.xy.cpu().numpy()

    if result.keypoints.conf is not None:
        kp_conf = result.keypoints.conf.cpu().numpy()
    else:
        kp_conf = np.ones(
            xy.shape[:2],
            dtype=np.float32
        )

    if result.boxes is not None:
        person_conf = (
            result.boxes.conf.cpu().numpy()
        )
    else:
        person_conf = np.ones(
            len(xy),
            dtype=np.float32
        )

    for person_id in range(len(xy)):

        keypoints = []

        for kp_idx in range(xy.shape[1]):

            x = float(xy[person_id, kp_idx, 0])
            y = float(xy[person_id, kp_idx, 1])

            score = float(
                kp_conf[person_id, kp_idx]
            )

            keypoints.append({
                "id": kp_idx + 1,
                "x": x,
                "y": y,
                "score": score,
            })

        persons.append({
            "id": person_id,
            "score": float(
                person_conf[person_id]
            ),
            "keypoints": keypoints,
        })

    return persons


def process_camera(
    model,
    camera_id,
    frames_dir,
    calibration_path,
    output_path,
    model_device,
):

    calib = load_calibration(
        calibration_path
    )

    camera_dir = os.path.join(
        frames_dir,
        f"Camera{camera_id}"
    )

    image_paths = sorted([
        os.path.join(camera_dir, f)
        for f in os.listdir(camera_dir)
        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )
    ])

    print(
        f"Camera {camera_id}: "
        f"{len(image_paths)} frames"
    )

    output = {
        "camera_id": camera_id,
        "frames": {}
    }

    for frame_idx, image_path in enumerate(
        image_paths
    ):

        frame = cv2.imread(image_path)

        if frame is None:
            continue

        frame = undistort_frame(
            frame,
            calib
        )

        result = model.predict(
            frame,
            device=model_device,
            verbose=False,

            # pode aumentar para melhorar precisão
            imgsz=1280,

            conf=0.20,

            # sem augment para manter comportamento
            # mais previsível inicialmente
            augment=False,
        )[0]

        persons = extract_persons(result)

        output["frames"][str(frame_idx)] = persons

        if frame_idx % 100 == 0:
            print(
                f"Cam {camera_id} | "
                f"frame {frame_idx}/{len(image_paths)} | "
                f"pessoas={len(persons)}"
            )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False
        )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="yolo26m-pose.pt"
    )

    parser.add_argument(
        "--frames",
        required=True
    )

    parser.add_argument(
        "--calib",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    parser.add_argument(
        "--cameras",
        default="0,1,2,3,4"
    )

    parser.add_argument(
        "--device",
        default="0"
    )

    args = parser.parse_args()

    camera_ids = [
        int(v)
        for v in args.cameras.split(",")
    ]

    model = YOLO(args.model)

    for camera_id in camera_ids:

        calibration_path = os.path.join(
            args.calib,
            f"calib_rt{camera_id}.npz"
        )

        output_path = os.path.join(
            args.output,
            f"camera_{camera_id}.json"
        )

        process_camera(
            model=model,
            camera_id=camera_id,
            frames_dir=args.frames,
            calibration_path=calibration_path,
            output_path=output_path,
            model_device=args.device,
        )


if __name__ == "__main__":
    main()