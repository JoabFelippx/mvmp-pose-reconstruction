"""
config.py — Carrega configuração com prioridade:
  1. Variável de ambiente
  2. config.json (fallback)

Uso:
    from config import cfg
    print(cfg.broker_uri)
    print(cfg.num_cameras)
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any

# ── helpers ────────────────────────────────────────────────────────────────────

def _env(key: str, fallback: Any, cast=str) -> Any:
    val = os.environ.get(key)
    if val is not None:
        if cast is bool:
            return val.lower() in ("1", "true", "yes")
        return cast(val)
    return fallback

def _load_json(path: str = "etc/config.json") -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ── dataclass de configuração ─────────────────────────────────────────────────

@dataclass
class AppConfig:
    # ── IS / broker ───────────────────────────────────────────────────────────
    broker_uri: str = "amqp://guest:guest@localhost:5672"
    use_is: bool = False
    is_calib_path: str = "calibrations/calib_rt"   # prefixo específico do modo IS; sufixo: {i}.npz
    is_num_cameras: int = 4

    # ── câmeras (modo dataset) ─────────────────────────────────────────────────
    num_cameras: int = 4
    calib_path: str = "calibrations/calib_rt"   # prefixo; sufixo: {i}.npz
    use_undistorted: bool = True
    apply_undistort: bool = True

    # ── keypoints ─────────────────────────────────────────────────────────────
    num_keypoints: int = 18

    # ── matcher ───────────────────────────────────────────────────────────────
    use_cycle_consistency: bool = True
    sigma_tolerance: float = 15.0
    max_error_per_joint: float = 20.0
    weight_quality: float = 0.7
    weight_quantity: float = 0.3
    min_compatibility_score: float = 0.3
    top_k_cycle_candidates: int = 3
    early_stop_cycle_score: float = 0.7
    weight_cycle: float = 0.5
    min_cycle_score: float = 0.50
    max_intersection_dist: float = 3.0
    weight_distance: float = 0.60
    weight_score: float = 0.40
    min_keypoints_for_grouping: int = 10
    min_kp_ratio: float = 0.30
    min_fallback_score: float = 0.78

    # ── campos estáticos (só config.json, não faz sentido como env var) ───────
    kp_weights: dict = field(default_factory=lambda: {str(i): 1.0 for i in range(17)})
    skeleton_connections: list = field(default_factory=list)
    keypoint_names: list = field(default_factory=list)


def load_config(json_path: str = "etc/config.json") -> AppConfig:
    raw = _load_json(json_path)

    ini  = raw.get("default_initialization", {})
    mat  = raw.get("matcher_parameters", {})
    kpts = raw.get("keypoint_settings", {})
    is_  = raw.get("is_settings", {})

    return AppConfig(
        # broker / IS
        broker_uri=_env("BROKER_URI",
                        is_.get("broker_uri", "amqp://guest:guest@localhost:5672")),
        use_is=_env("USE_IS",
                    is_.get("use_is", False), bool),
        is_calib_path=_env("IS_CALIB_PATH",
                           is_.get("calib_path", ini.get("calib_path", "calibrations/calib_rt"))),
        is_num_cameras=_env("IS_NUM_CAMERAS",
                            is_.get("num_cameras", ini.get("num_cameras", 4)), int),

        # câmeras
        num_cameras=_env("NUM_CAMERAS",
                         ini.get("num_cameras", 4), int),
        calib_path=_env("CALIB_PATH",
                        ini.get("calib_path", "calibrations/calib_rt")),
        use_undistorted=_env("USE_UNDISTORTED",
                             ini.get("use_undistorted", True), bool),
        apply_undistort=_env("APPLY_UNDISTORT",
                             ini.get("apply_undistort", True), bool),

        # keypoints
        num_keypoints=_env("NUM_KEYPOINTS",
                           kpts.get("num_keypoints", 18), int),

        # matcher — todos os floats/ints simples viram env var
        use_cycle_consistency=_env("MATCHER_USE_CYCLE_CONSISTENCY",
                                   mat.get("use_cycle_consistency", True), bool),
        sigma_tolerance=_env("MATCHER_SIGMA_TOLERANCE",
                             mat.get("sigma_tolerance", 15.0), float),
        max_error_per_joint=_env("MATCHER_MAX_ERROR_PER_JOINT",
                                 mat.get("max_error_per_joint", 20.0), float),
        weight_quality=_env("MATCHER_WEIGHT_QUALITY",
                            mat.get("weight_quality", 0.7), float),
        weight_quantity=_env("MATCHER_WEIGHT_QUANTITY",
                             mat.get("weight_quantity", 0.3), float),
        min_compatibility_score=_env("MATCHER_MIN_COMPATIBILITY_SCORE",
                                     mat.get("min_compatibility_score", 0.3), float),
        top_k_cycle_candidates=_env("MATCHER_TOP_K_CYCLE_CANDIDATES",
                                    mat.get("top_k_cycle_candidates", 3), int),
        early_stop_cycle_score=_env("MATCHER_EARLY_STOP_CYCLE_SCORE",
                                    mat.get("early_stop_cycle_score", 0.7), float),
        weight_cycle=_env("MATCHER_WEIGHT_CYCLE",
                          mat.get("weight_cycle", 0.5), float),
        min_cycle_score=_env("MATCHER_MIN_CYCLE_SCORE",
                             mat.get("min_cycle_score", 0.50), float),
        max_intersection_dist=_env("MATCHER_MAX_INTERSECTION_DIST",
                                   mat.get("max_intersection_dist", 3.0), float),
        weight_distance=_env("MATCHER_WEIGHT_DISTANCE",
                             mat.get("weight_distance", 0.60), float),
        weight_score=_env("MATCHER_WEIGHT_SCORE",
                          mat.get("weight_score", 0.40), float),
        min_keypoints_for_grouping=_env("MATCHER_MIN_KEYPOINTS_FOR_GROUPING",
                                        mat.get("min_keypoints_for_grouping", 10), int),
        min_kp_ratio=_env("MATCHER_MIN_KP_RATIO",
                          mat.get("min_kp_ratio", 0.30), float),
        min_fallback_score=_env("MATCHER_MIN_FALLBACK_SCORE",
                                mat.get("min_fallback_score", 0.78), float),

        # estáticos — apenas config.json
        kp_weights=mat.get("kp_weights", {str(i): 1.0 for i in range(17)}),
        skeleton_connections=kpts.get("skeleton_connections", []),
        keypoint_names=kpts.get("keypoint_names", []),
    )


# Singleton — importe de qualquer módulo com: from config import cfg
cfg = load_config()