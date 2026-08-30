import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_calibration(path):
    data = np.load(path)

    return {
        "K": data["K"],
        "nK": data["nK"],
        "dist": data["dist"],
        "roi": data["roi"] if "roi" in data else None,
    }


def list_camera_images(frames_dir, camera_id):
    camera_dir = os.path.join(
        frames_dir,
        f"Camera{camera_id}"
    )

    if not os.path.isdir(camera_dir):
        raise FileNotFoundError(
            f"Pasta da Camera {camera_id} nao encontrada: {camera_dir}"
        )

    image_paths = sorted([
        os.path.join(camera_dir, filename)
        for filename in os.listdir(camera_dir)
        if filename.lower().endswith(IMAGE_EXTENSIONS)
    ])

    return image_paths


def build_undistort_maps(calib, image_path):
    """
    Calcula os mapas UMA VEZ por camera.

    O codigo antigo chamava cv2.initUndistortRectifyMap()
    para todo frame, o que gera um custo de CPU desnecessario.
    """
    sample = cv2.imread(image_path)

    if sample is None:
        raise RuntimeError(
            f"Nao foi possivel ler a imagem para criar os mapas: {image_path}"
        )

    h, w = sample.shape[:2]

    map1, map2 = cv2.initUndistortRectifyMap(
        calib["K"],
        calib["dist"],
        None,
        calib["nK"],
        (w, h),
        cv2.CV_16SC2,
    )

    return map1, map2


def prepare_frame(task):
    """
    Leitura + undistortion executadas nas threads de CPU.

    Retorna:
        camera_id, frame_idx, frame
    """
    (
        camera_id,
        frame_idx,
        image_path,
        map1,
        map2,
        roi,
    ) = task

    frame = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return camera_id, frame_idx, None

    frame = cv2.remap(
        frame,
        map1,
        map2,
        cv2.INTER_LINEAR,
    )

    # Mantem exatamente o comportamento do script original.
    if roi is not None:
        x, y, w_roi, h_roi = [int(v) for v in roi]
        frame = frame[0:h_roi, 0:w_roi]

    return camera_id, frame_idx, frame


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
            dtype=np.float32,
        )

    if result.boxes is not None:
        person_conf = (
            result.boxes.conf
            .cpu()
            .numpy()
        )
    else:
        person_conf = np.ones(
            len(xy),
            dtype=np.float32,
        )

    for person_id in range(len(xy)):
        keypoints = []

        for kp_idx in range(xy.shape[1]):
            x = float(
                xy[person_id, kp_idx, 0]
            )
            y = float(
                xy[person_id, kp_idx, 1]
            )
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


def save_outputs(outputs, output_dir):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    for camera_id, output in outputs.items():
        # Ordena numericamente os frames antes de salvar.
        frames_sorted = dict(
            sorted(
                output["frames"].items(),
                key=lambda item: int(item[0]),
            )
        )

        data = {
            "camera_id": camera_id,
            "frames": frames_sorted,
        }

        output_path = os.path.join(
            output_dir,
            f"camera_{camera_id}.json",
        )

        temp_path = (
            output_path + ".tmp"
        )

        with open(
            temp_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
            )

        os.replace(
            temp_path,
            output_path,
        )


def build_camera_data(
    frames_dir,
    calib_dir,
    camera_ids,
):
    camera_data = {}

    for camera_id in camera_ids:
        calibration_path = os.path.join(
            calib_dir,
            f"calib_rt{camera_id}.npz",
        )

        if not os.path.exists(
            calibration_path
        ):
            raise FileNotFoundError(
                f"Calibracao da camera {camera_id} nao encontrada: "
                f"{calibration_path}"
            )

        image_paths = list_camera_images(
            frames_dir,
            camera_id,
        )

        print(
            f"Camera {camera_id}: "
            f"{len(image_paths)} frames"
        )

        if not image_paths:
            camera_data[camera_id] = {
                "paths": [],
                "calib": load_calibration(
                    calibration_path
                ),
                "map1": None,
                "map2": None,
            }
            continue

        calib = load_calibration(
            calibration_path
        )

        map1, map2 = (
            build_undistort_maps(
                calib,
                image_paths[0],
            )
        )

        camera_data[camera_id] = {
            "paths": image_paths,
            "calib": calib,
            "map1": map1,
            "map2": map2,
        }

    return camera_data


def build_tasks(
    camera_data,
    camera_ids,
):
    """
    Intercala cameras por frame:

        frame 0 cam0
        frame 0 cam1
        frame 0 cam2
        frame 1 cam0
        ...

    Assim um batch da GPU contem imagens de varias cameras.
    """
    max_frames = max(
        (
            len(
                camera_data[camera_id][
                    "paths"
                ]
            )
            for camera_id in camera_ids
        ),
        default=0,
    )

    tasks = []

    for frame_idx in range(max_frames):
        for camera_id in camera_ids:
            data = camera_data[
                camera_id
            ]

            paths = data["paths"]

            if frame_idx >= len(paths):
                continue

            tasks.append((
                camera_id,
                frame_idx,
                paths[frame_idx],
                data["map1"],
                data["map2"],
                data["calib"]["roi"],
            ))

    return tasks


def process_all(
    model,
    tasks,
    outputs,
    output_dir,
    device,
    workers,
    batch_size,
    imgsz,
    confidence,
    quantize,
    save_every,
):
    total_images = len(tasks)

    if total_images == 0:
        print("Nenhuma imagem encontrada.")
        return

    print()
    print(
        f"Imagens totais : {total_images}"
    )
    print(
        f"Threads CPU    : {workers}"
    )
    print(
        f"Batch GPU      : {batch_size}"
    )
    print(
        f"imgsz          : {imgsz}"
    )
    print(
        f"device         : {device}"
    )
    print(
        f"quantize       : {quantize}"
    )
    print()

    processed = 0
    start_time = time.perf_counter()

    # Evita multiplicar threads internas do OpenCV
    # por cada ThreadPoolExecutor worker.
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    with ThreadPoolExecutor(
        max_workers=workers
    ) as executor:

        for chunk_start in range(
            0,
            total_images,
            batch_size,
        ):
            chunk = tasks[
                chunk_start:
                chunk_start + batch_size
            ]

            # -------------------------------------------------
            # CPU: leitura + remap em paralelo
            # -------------------------------------------------

            prepared = list(
                executor.map(
                    prepare_frame,
                    chunk,
                )
            )

            valid_metadata = []
            valid_frames = []

            for (
                camera_id,
                frame_idx,
                frame,
            ) in prepared:

                if frame is None:
                    outputs[camera_id][
                        "frames"
                    ][str(frame_idx)] = []
                    continue

                valid_metadata.append(
                    (
                        camera_id,
                        frame_idx,
                    )
                )
                valid_frames.append(
                    frame
                )

            if not valid_frames:
                continue

            # -------------------------------------------------
            # GPU: UMA chamada batched da YOLO
            #
            # Nao executamos model.predict em varias threads,
            # evitando concorrencia no mesmo modelo/GPU.
            # -------------------------------------------------

            results = model.predict(
                source=valid_frames,
                device=device,
                verbose=False,
                imgsz=imgsz,
                conf=confidence,
                augment=False,
                quantize=quantize,
                batch=len(valid_frames),
                stream=False,
            )

            for (
                camera_id,
                frame_idx,
            ), result in zip(
                valid_metadata,
                results,
            ):
                persons = extract_persons(
                    result
                )

                outputs[camera_id][
                    "frames"
                ][str(frame_idx)] = persons

                processed += 1

            if (
                processed % 100 < len(valid_frames)
                or processed == total_images
            ):
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                fps = (
                    processed / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"[2D] "
                    f"{processed}/{total_images} "
                    f"| {fps:.2f} img/s"
                )

            if (
                save_every > 0
                and processed > 0
                and processed % save_every
                < len(valid_frames)
            ):
                save_outputs(
                    outputs,
                    output_dir,
                )

    save_outputs(
        outputs,
        output_dir,
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    fps = (
        processed / elapsed
        if elapsed > 0
        else 0.0
    )

    print()
    print(
        "Extracao concluida."
    )
    print(
        f"Processadas : {processed} imagens"
    )
    print(
        f"Tempo       : {elapsed:.2f} s"
    )
    print(
        f"Media       : {fps:.2f} img/s"
    )
    print(
        f"Saida       : {output_dir}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extracao YOLO Pose multi-camera "
            "com I/O em threads e inferencia GPU em batch."
        )
    )

    parser.add_argument(
        "--model",
        default="yolo26m-pose.pt",
    )

    parser.add_argument(
        "--frames",
        required=True,
    )

    parser.add_argument(
        "--calib",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--cameras",
        default="0,1,2,3,4",
    )

    parser.add_argument(
        "--device",
        default="0",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help=(
            "Numero de threads para leitura "
            "e undistortion."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Numero de imagens por chamada "
            "da YOLO. Reduza se faltar VRAM."
        ),
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--quantize",
        type=int,
        choices=[16, 32],
        default=16,
        help=(
            "Precisao da inferencia: 16=FP16 "
            "(mais rapido na GPU), 32=FP32."
        ),
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=500,
        help=(
            "Salva checkpoints dos JSONs "
            "a cada N imagens. 0 desativa."
        ),
    )

    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError(
            "--workers precisa ser >= 1"
        )

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size precisa ser >= 1"
        )

    camera_ids = [
        int(value.strip())
        for value in args.cameras.split(",")
        if value.strip()
    ]

    print(
        "Carregando modelo:"
    )
    print(
        f"  {args.model}"
    )

    model = YOLO(
        args.model
    )

    camera_data = build_camera_data(
        frames_dir=args.frames,
        calib_dir=args.calib,
        camera_ids=camera_ids,
    )

    outputs = {
        camera_id: {
            "camera_id": camera_id,
            "frames": {},
        }
        for camera_id in camera_ids
    }

    tasks = build_tasks(
        camera_data,
        camera_ids,
    )

    process_all(
        model=model,
        tasks=tasks,
        outputs=outputs,
        output_dir=args.output,
        device=args.device,
        workers=args.workers,
        batch_size=args.batch_size,
        imgsz=args.imgsz,
        confidence=args.conf,
        quantize=args.quantize,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
