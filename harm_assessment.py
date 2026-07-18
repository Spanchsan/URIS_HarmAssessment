import sys
import transformers.utils.versions as transformers_versions
original_require_version_core = transformers_versions.require_version_core

def patched_require_version_core(requirement, hint=None):
    if "numpy" in requirement:
        return # Skip the broken check for numpy entirely
    return original_require_version_core(requirement, hint)

transformers_versions.require_version_core = patched_require_version_core

import torch
import torch.distributed as dist
if not hasattr(dist, 'ReduceOp'):
    class MockReduceOp:
        SUM = 'SUM'
        PRODUCT = 'PRODUCT'
        MIN = 'MIN'
        MAX = 'MAX'
        BAND = 'BAND'
        BOR = 'BOR'
        BXOR = 'BXOR'
        AVG = 'AVG'
    dist.ReduceOp = MockReduceOp

import argparse
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from model import GraphModel 
from functools import wraps

import os

import threading
import queue
import time
import requests
import base64
import gc
os.environ["TRITON_ALWAYS_COMPILE"] = "0"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"
os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"


NV_API_KEY = os.getenv('NVIDIA_API_KEY', '')


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


NTU_NAMES = [
    "SpineBase",       # 0
    "SpineMid",        # 1
    "Neck",            # 2
    "Head",            # 3
    "ShoulderLeft",    # 4
    "ElbowLeft",       # 5
    "WristLeft",       # 6
    "HandLeft",        # 7
    "ShoulderRight",   # 8
    "ElbowRight",      # 9
    "WristRight",      # 10
    "HandRight",       # 11
    "HipLeft",         # 12
    "KneeLeft",        # 13
    "AnkleLeft",       # 14
    "FootLeft",        # 15
    "HipRight",        # 16
    "KneeRight",       # 17
    "AnkleRight",      # 18
    "FootRight",       # 19
    "SpineShoulder",   # 20
    "HandTipLeft",     # 21
    "ThumbLeft",       # 22
    "HandTipRight",    # 23
    "ThumbRight",      # 24
]

labels = [
    'drink water', 'eat meal/snack', 'brushing teeth', 'brushing hair', 'drop', 'pickup',
    'throw', 'sitting down', 'standing up', 'clapping', 'reading', 'writing',
    'tear up paper', 'wear jacket', 'take off jacket', 'wear a shoe', 'take off a shoe',
    'wear on glasses', 'take off glasses', 'put on a hat/cap', 'take off a hat/cap',
    'cheer up', 'hand waving', 'kicking something', 'reach into pocket', 'hopping',
    'jump up', 'make a phone call/answer phone', 'playing with phone/tablet', 'typing on a keyboard',
    'pointing to something with finger', 'taking a selfie', 'check time (from watch)',
    'rub two hands together', 'nod head/bow', 'shake head', 'wipe face', 'salute',
    'put the palms together', 'sneeze/cough', 'staggering', 'falling', 'touch head (headache)',
    'touch chest (stomachache/heart pain)', 'touch back (backache)', 'touch neck (neckache)',
    'nausea or vomiting condition', 'use a fan (with hand or paper)/feeling warm',
    'punching/slapping other person', 'kicking other person', 'pushing other person',
    'pat on back of other person', 'point finger at the other person', 'hugging other person',
    'giving something to other person', 'touch other person\'s pocket', 'handshaking',
    'walking towards each other', 'walking apart from each other'
]

TARGET_IDS = [41, 42, 44, 45, 46, 47]

PROMPT = '''
You are a medical expert. Analyze the following image and provide a detailed description if medical help is required, including any potential concerns that may be relevant. Please be thorough and specific in your analysis.
If person is fallen down and is in dangerous position (face down or face up), output "MEDICAL EMERGENCY".
If person is having heart ache or symptoms of chest pain, output "MEDICAL EMERGENCY".
If person is fallen down but is in safe position (on the side), output "MEDICAL ATTENTION".
If person is not fallen down, but there are any other medical concerns, output "MEDICAL ATTENTION".
Otherwise, output "NO MEDICAL CONCERNS".
DO NOT output anything else. Only output one of the three phrases above.
'''

# NTU skeleton bones used for 3D fitting.
NTU_BONES = [
    (0, 1), (1, 20), (20, 2), (2, 3),

    (20, 4), (4, 5), (5, 6), (6, 7),
    (7, 21), (6, 22),

    (20, 8), (8, 9), (9, 10), (10, 11),
    (11, 23), (10, 24),

    (0, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19),

    # stabilizers
    (4, 8),     # shoulder width
    (12, 16),   # hip width
]


def time_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"'{func.__name__}' took {end_time - start_time:.6f} seconds")
        return result
    return wrapper

# Approx adult bone lengths in meters.
def make_default_bone_lengths(body_height_m=1.70):
    s = body_height_m / 1.70

    return {
        (0, 1): 0.25 * s,
        (1, 20): 0.30 * s,
        (20, 2): 0.08 * s,
        (2, 3): 0.16 * s,

        (20, 4): 0.18 * s,
        (4, 5): 0.30 * s,
        (5, 6): 0.27 * s,
        (6, 7): 0.10 * s,
        (7, 21): 0.08 * s,
        (6, 22): 0.08 * s,

        (20, 8): 0.18 * s,
        (8, 9): 0.30 * s,
        (9, 10): 0.27 * s,
        (10, 11): 0.10 * s,
        (11, 23): 0.08 * s,
        (10, 24): 0.08 * s,

        (0, 12): 0.14 * s,
        (12, 13): 0.45 * s,
        (13, 14): 0.43 * s,
        (14, 15): 0.12 * s,

        (0, 16): 0.14 * s,
        (16, 17): 0.45 * s,
        (17, 18): 0.43 * s,
        (18, 19): 0.12 * s,

        (4, 8): 0.36 * s,
        (12, 16): 0.28 * s,
    }


class WholeBodyToNTU25Mapper:
    """
    Converts MMPose COCO-WholeBody keypoints to NTU RGB+D 25-joint 2D layout.

    COCO-WholeBody approximate layout:
        0-16       COCO body
        17-22      feet
        23-90      face
        91-111     left hand
        112-132    right hand

    Body COCO:
        0 nose
        1 left_eye
        2 right_eye
        3 left_ear
        4 right_ear
        5 left_shoulder
        6 right_shoulder
        7 left_elbow
        8 right_elbow
        9 left_wrist
        10 right_wrist
        11 left_hip
        12 right_hip
        13 left_knee
        14 right_knee
        15 left_ankle
        16 right_ankle

    Feet:
        17 left_big_toe
        18 left_small_toe
        19 left_heel
        20 right_big_toe
        21 right_small_toe
        22 right_heel

    Hand local order:
        0 wrist
        1-4 thumb
        5-8 index
        9-12 middle
        13-16 ring
        17-20 pinky
    """

    def __init__(self, conf_thr=0.20):
        self.conf_thr = conf_thr
        self.prev_ntu2d = None
        self.prev_score = None

    def __call__(self, wholebody_kpts):
        """
        Args:
            wholebody_kpts: np.ndarray [K, 3], K can be 17 (COCO), 26 (Halpe/Pose26), or 133 (wholebody).
                            columns are x, y, score.

        Returns:
            ntu2d: np.ndarray [25, 2]
            score: np.ndarray [25]
        """

        k = np.asarray(wholebody_kpts, dtype=np.float32)
        K = k.shape[0]

        ntu = np.zeros((25, 2), dtype=np.float32)
        score = np.zeros((25,), dtype=np.float32)

        def valid(src):
            return src < K and k[src, 2] >= self.conf_thr

        def set_joint(dst, src, mult=1.0):
            if valid(src):
                ntu[dst] = k[src, :2]
                score[dst] = k[src, 2] * mult

        def avg_from_src(dst, srcs, mult=1.0):
            pts = []
            ws = []
            confs = []
            for src in srcs:
                if valid(src):
                    c = float(k[src, 2])
                    pts.append(k[src, :2] * c)
                    ws.append(c)
                    confs.append(c)
            if len(pts) == 0:
                return
            ntu[dst] = np.sum(pts, axis=0) / max(np.sum(ws), 1e-6)
            score[dst] = float(np.mean(confs)) * mult

        def avg_from_ntu(dst, srcs, mult=1.0):
            pts = []
            ws = []
            confs = []
            for src in srcs:
                if score[src] >= self.conf_thr:
                    c = float(score[src])
                    pts.append(ntu[src] * c)
                    ws.append(c)
                    confs.append(c)
            if len(pts) == 0:
                return
            ntu[dst] = np.sum(pts, axis=0) / max(np.sum(ws), 1e-6)
            score[dst] = float(np.mean(confs)) * mult

        def extrapolate(dst, a, b, ratio, mult=0.7):
            if score[a] >= self.conf_thr and score[b] >= self.conf_thr:
                ntu[dst] = ntu[b] + ratio * (ntu[b] - ntu[a])
                score[dst] = min(score[a], score[b]) * mult

        # ----------------------------------------------------
        # Detect input format
        # ----------------------------------------------------
        is_halpe26 = K == 26

        # ----------------------------------------------------
        # Direct body mapping
        # ----------------------------------------------------

        if is_halpe26:
            # Halpe26 / Pose26 standard ordering
            set_joint(4, 5)   # Left Shoulder
            set_joint(8, 6)   # Right Shoulder

            set_joint(5, 7)   # Left Elbow
            set_joint(9, 8)   # Right Elbow

            set_joint(6, 9)   # Left Wrist
            set_joint(10, 10) # Right Wrist

            set_joint(12, 11) # Left Hip
            set_joint(16, 12) # Right Hip

            set_joint(13, 13) # Left Knee
            set_joint(17, 14) # Right Knee

            set_joint(14, 15) # Left Ankle
            set_joint(18, 16) # Right Ankle

            # Neck / SpineShoulder
            set_joint(20, 0)  # Often nose or neck in some variants — adjust if needed
        else:
            # Original COCO-17 + WholeBody logic
            set_joint(4, 5)
            set_joint(8, 6)
            set_joint(5, 7)
            set_joint(9, 8)
            set_joint(6, 9)
            set_joint(10, 10)
            set_joint(12, 11)
            set_joint(16, 12)
            set_joint(13, 13)
            set_joint(17, 14)
            set_joint(14, 15)
            set_joint(18, 16)

        # ----------------------------------------------------
        # Spine and head (common for both)
        # ----------------------------------------------------

        avg_from_ntu(20, [4, 8])      # SpineShoulder = shoulder midpoint
        avg_from_ntu(0, [12, 16])     # SpineBase = hip midpoint

        if score[0] > 0 and score[20] > 0:
            ntu[1] = 0.55 * ntu[0] + 0.45 * ntu[20]   # Mid-spine
            score[1] = min(score[0], score[20]) * 0.95

        # Head (use nose if available)
        if is_halpe26:
            set_joint(3, 0)           # Head ≈ Nose
        else:
            avg_from_src(3, [0, 1, 2, 3, 4])  # Original head logic

        # Neck
        if score[20] > 0 and score[3] > 0:
            ntu[2] = 0.65 * ntu[20] + 0.35 * ntu[3]
            score[2] = min(score[20], score[3]) * 0.95

        # ----------------------------------------------------
        # Hands
        # ----------------------------------------------------

        if K >= 133 or is_halpe26:   # Halpe26 usually doesn't have full hands, so fallback
            # For Halpe26 we use wrist extrapolation
            if is_halpe26:
                extrapolate(7, 5, 6, ratio=0.40)      # HandLeft
                extrapolate(11, 9, 10, ratio=0.40)    # HandRight
                extrapolate(21, 6, 7, ratio=0.75, mult=0.6)
                extrapolate(23, 10, 11, ratio=0.75, mult=0.6)
                self._fake_thumb(ntu, score, wrist=6, hand=7, dst=22, side="left")
                self._fake_thumb(ntu, score, wrist=10, hand=11, dst=24, side="right")
            else:
                # Full wholebody hand logic (unchanged)
                LH = 91
                RH = 112
                avg_from_src(7, [LH + 0, LH + 5, LH + 9, LH + 13, LH + 17])
                avg_from_src(11, [RH + 0, RH + 5, RH + 9, RH + 13, RH + 17])
                set_joint(21, LH + 12)
                set_joint(23, RH + 12)
                set_joint(22, LH + 4)
                set_joint(24, RH + 4)

                if score[7] < self.conf_thr:
                    extrapolate(7, 5, 6, ratio=0.35)
                if score[11] < self.conf_thr:
                    extrapolate(11, 9, 10, ratio=0.35)
                # ... rest of hand logic unchanged
        else:
            # COCO-17 fallback
            extrapolate(7, 5, 6, ratio=0.35)
            extrapolate(11, 9, 10, ratio=0.35)
            extrapolate(21, 6, 7, ratio=0.70, mult=0.5)
            extrapolate(23, 10, 11, ratio=0.70, mult=0.5)
            self._fake_thumb(ntu, score, wrist=6, hand=7, dst=22, side="left")
            self._fake_thumb(ntu, score, wrist=10, hand=11, dst=24, side="right")

        # ----------------------------------------------------
        # Feet
        # ----------------------------------------------------

        if K >= 23 or is_halpe26:
            if is_halpe26:
                # Halpe26 has basic foot points
                avg_from_src(15, [17, 18])   # FootLeft (adjust indices if your Halpe has toes)
                avg_from_src(19, [19, 20])   # FootRight
            else:
                avg_from_src(15, [17, 18, 19])
                avg_from_src(19, [20, 21, 22])

            if score[15] < self.conf_thr:
                extrapolate(15, 13, 14, ratio=0.30)
            if score[19] < self.conf_thr:
                extrapolate(19, 17, 18, ratio=0.30)
        else:
            extrapolate(15, 13, 14, ratio=0.30)
            extrapolate(19, 17, 18, ratio=0.30)

        # ----------------------------------------------------
        # Final cleanup
        # ----------------------------------------------------
        ntu, score = self.fill_missing(ntu, score)

        self.prev_ntu2d = ntu.copy()
        self.prev_score = score.copy()

        return ntu, score


    def _fake_thumb(self, ntu, score, wrist, hand, dst, side):
        if score[wrist] < self.conf_thr or score[hand] < self.conf_thr:
            return

        v = ntu[hand] - ntu[wrist]
        n = np.linalg.norm(v)
        if n < 1e-6:
            return

        v = v / n
        perp = np.array([-v[1], v[0]], dtype=np.float32)

        sign = 1.0 if side == "left" else -1.0

        ntu[dst] = ntu[hand] + 20.0 * v + sign * 20.0 * perp
        score[dst] = min(score[wrist], score[hand]) * 0.35

    def fill_missing(self, ntu, score):
        parents = {
            0: 1,
            1: 20,
            2: 20,
            3: 2,

            4: 20,
            5: 4,
            6: 5,
            7: 6,
            21: 7,
            22: 6,

            8: 20,
            9: 8,
            10: 9,
            11: 10,
            23: 11,
            24: 10,

            12: 0,
            13: 12,
            14: 13,
            15: 14,

            16: 0,
            17: 16,
            18: 17,
            19: 18,

            20: 1,
        }

        # center fallback
        center = None
        if score[0] >= self.conf_thr:
            center = ntu[0]
        elif score[20] >= self.conf_thr:
            center = ntu[20]
        else:
            valid_pts = ntu[score >= self.conf_thr]
            if len(valid_pts) > 0:
                center = np.mean(valid_pts, axis=0)
            else:
                center = np.array([0.0, 0.0], dtype=np.float32)

        for j in range(25):
            if score[j] >= self.conf_thr:
                continue

            # previous frame is best
            if self.prev_ntu2d is not None and self.prev_score is not None:
                if self.prev_score[j] >= self.conf_thr:
                    ntu[j] = self.prev_ntu2d[j]
                    score[j] = self.prev_score[j] * 0.50
                    continue

            # parent fallback
            p = parents.get(j, None)
            if p is not None and score[p] >= self.conf_thr:
                ntu[j] = ntu[p]
                score[j] = score[p] * 0.25
                continue

            # center fallback
            ntu[j] = center
            score[j] = 0.01

        return ntu, score


class NTU25Pseudo3DFitter:
    """
    Converts NTU-25 2D joints into approximate Kinect-style 3D joints.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        body_height_m=1.70,
        conf_thr=0.20,
        flip_y=True,
        root_depth_weight=1.5,
        temporal_depth_weight=0.4,
        ema_alpha=0.75,
    ):
        self.K = intrinsics
        self.conf_thr = conf_thr
        self.flip_y = flip_y
        self.root_depth_weight = root_depth_weight
        self.temporal_depth_weight = temporal_depth_weight
        self.ema_alpha = ema_alpha

        self.bone_len = make_default_bone_lengths(body_height_m)
        self.prev_z = None
        self.prev_xyz = None

        # Precompute bone pairs
        self.bone_pairs = np.array(NTU_BONES, dtype=np.int32)  # (N_bones, 2)

    def reset(self):
        self.prev_z = None
        self.prev_xyz = None

    def estimate_initial_root_depth(self, ntu2d, score):
        candidates = []
        if score[4] >= self.conf_thr and score[8] >= self.conf_thr:
            pix = np.linalg.norm(ntu2d[4] - ntu2d[8])
            if pix > 5:
                candidates.append(self.bone_len[(4, 8)] * self.K.fx / pix)

        if score[12] >= self.conf_thr and score[16] >= self.conf_thr:
            pix = np.linalg.norm(ntu2d[12] - ntu2d[16])
            if pix > 5:
                candidates.append(self.bone_len[(12, 16)] * self.K.fx / pix)

        return float(np.median(candidates)) if candidates else 2.5

    def pixels_to_camera(self, uv, z):
        x = (uv[:, 0] - self.K.cx) * z / self.K.fx
        if self.flip_y:
            y = -(uv[:, 1] - self.K.cy) * z / self.K.fy
        else:
            y = (uv[:, 1] - self.K.cy) * z / self.K.fy
        return np.stack([x, y, z], axis=-1).astype(np.float32)

    def _residual_and_jac(self, z, ntu2d, score, z0, bone_a, bone_b, bone_L, prev_z):
        """Compute residual and analytical Jacobian w.r.t. depths z"""
        z = np.clip(z, 0.3, 10.0)
        xyz = self.pixels_to_camera(ntu2d, z)

        res_list = []

        # Bone length residuals
        if len(bone_L) > 0:
            diffs = xyz[bone_a] - xyz[bone_b]
            dists = np.linalg.norm(diffs, axis=1)
            res_list.append((dists - bone_L) * 5.0)

        # Root depth constraint
        res_list.append(np.array([(z[0] - z0) * self.root_depth_weight], dtype=np.float32))

        # Depth spread regularization
        res_list.append((z - z[0]) * 0.03)

        # Temporal smoothness
        if prev_z is not None:
            res_list.append((z - prev_z) * self.temporal_depth_weight)

        res = np.concatenate(res_list)

        # --- Jacobian (n_res, 25) ---
        n_bones = len(bone_L)
        n_res = n_bones + 1 + 25 + (25 if prev_z is not None else 0)
        J = np.zeros((n_res, 25), dtype=np.float32)

        row = 0

        # Bone length Jacobian w.r.t. depth z
        for i in range(n_bones):
            a = bone_a[i]
            b = bone_b[i]
            diff = xyz[a] - xyz[b]
            dist = dists[i]   # reuse from above

            if dist > 1e-8:
                # Direction vector (unit)
                unit = diff / dist
                
                # How changing z[a] and z[b] affects x,y,z coordinates
                # Note: x and y depend on z through projection
                scale_a = z[a] / self.K.fx if z[a] > 0 else 0
                scale_b = z[b] / self.K.fx if z[b] > 0 else 0

                # Partial derivative of distance w.r.t. z_a and z_b
                # This is an approximation focusing on depth effect
                J[row, a] = np.dot(unit, [scale_a * (ntu2d[a,0]-self.K.cx)/z[a] if z[a]>0 else 0,
                                         scale_a * (ntu2d[a,1]-self.K.cy)/z[a] if z[a]>0 else 0,  # y already flipped
                                         1.0]) * 5.0
                
                J[row, b] = np.dot(unit, [-scale_b * (ntu2d[b,0]-self.K.cx)/z[b] if z[b]>0 else 0,
                                         -scale_b * (ntu2d[b,1]-self.K.cy)/z[b] if z[b]>0 else 0,
                                         -1.0]) * 5.0

            row += 1

        # Root constraint Jacobian
        J[row, 0] = self.root_depth_weight
        row += 1

        # Depth spread Jacobian
        J[row:row+25, :] = 0.03
        J[row:row+25, 0] -= 0.03
        row += 25

        # Temporal Jacobian
        if prev_z is not None:
            J[row:row+25] = self.temporal_depth_weight

        return res, J
    

    def lift(self, ntu2d, score):
        z0 = self.estimate_initial_root_depth(ntu2d, score)
        z0 = float(np.clip(z0, 0.3, 10.0))

        # Initialize depths
        if self.prev_z is None:
            init_z = np.full(25, z0, dtype=np.float32)
        else:
            init_z = self.prev_z.copy()
            init_z[score < self.conf_thr] = z0

        # Filter valid bones
        valid_mask = (score[self.bone_pairs[:, 0]] >= self.conf_thr) & \
                     (score[self.bone_pairs[:, 1]] >= self.conf_thr)
        valid_idx = np.where(valid_mask)[0]

        bone_a = self.bone_pairs[valid_idx, 0]
        bone_b = self.bone_pairs[valid_idx, 1]
        bone_L = np.array([self.bone_len[(int(a), int(b))] for a, b in self.bone_pairs[valid_idx]])

        # Closure for least_squares
        def residual(z):
            res, _ = self._residual_and_jac(z, ntu2d, score, z0, bone_a, bone_b, bone_L, self.prev_z)
            return res

        def jac(z):
            _, J = self._residual_and_jac(z, ntu2d, score, z0, bone_a, bone_b, bone_L, self.prev_z)
            return J

        # Solve with analytical Jacobian
        result = least_squares(
            residual,
            init_z,
            jac=jac,                  
            bounds=(0.3, 10.0),
            max_nfev=50,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
            verbose=0,
            method='trf'
        )

        z = result.x.astype(np.float32)
        xyz = self.pixels_to_camera(ntu2d, z)

        # Low confidence + EMA smoothing
        if self.prev_xyz is not None:
            low = score < self.conf_thr
            xyz[low] = self.prev_xyz[low]
            xyz = self.ema_alpha * xyz + (1.0 - self.ema_alpha) * self.prev_xyz

        self.prev_z = z.copy()
        self.prev_xyz = xyz.copy()

        return xyz


class NTUModelPreprocessor:
    """
    Produces clean model input.

    Default output:
        tensor shape [1, 3, T, 25, 1]

    Many ST-GCN/2s-AGCN/CTR-GCN style NTU models use:
        N, C, T, V, M

    where:
        N = batch
        C = 3 xyz
        T = frames
        V = 25 joints
        M = people, here 1
    """

    def __init__(
        self,
        target_frames=64,
        root_center=True,
        root_joint=20,
        scale_normalize=False,
        smooth_window=7,
    ):
        self.target_frames = target_frames
        self.root_center = root_center
        self.root_joint = root_joint
        self.scale_normalize = scale_normalize
        self.smooth_window = smooth_window

    def __call__(self, seq_xyz):
        """
        Args:
            seq_xyz: [T_raw, 25, 3]

        Returns:
            model_input: [1, 3, T, 25, 1]
            clean_seq: [T, 25, 3]
        """

        seq = np.asarray(seq_xyz, dtype=np.float32)

        # Temporal interpolation to fixed T
        seq = self.interpolate_time(seq, self.target_frames)

        seq = self.smooth(seq)

        # Root centering, important to center along joint number 20 as in feeder of the model
        if self.root_center:
            seq = seq - seq[:, self.root_joint:self.root_joint + 1, :]

        if self.scale_normalize:
            seq = self.normalize_scale(seq)

        # fill NaNs
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        model_input = seq.transpose(2, 0, 1)         
        model_input = model_input[None, :, :, :, None]  # [1, 3, T, 25, 1]

        return model_input.astype(np.float32), seq.astype(np.float32)

    @staticmethod
    def interpolate_time(seq, target_frames):
        T, V, C = seq.shape

        if T == target_frames:
            return seq.copy()

        if T <= 1:
            return np.repeat(seq, target_frames, axis=0)

        old_idx = np.linspace(0.0, 1.0, T)
        new_idx = np.linspace(0.0, 1.0, target_frames)

        out = np.zeros((target_frames, V, C), dtype=np.float32)

        for v in range(V):
            for c in range(C):
                out[:, v, c] = np.interp(new_idx, old_idx, seq[:, v, c])

        return out

    def smooth(self, seq):
        T = seq.shape[0]

        if T < 5:
            return seq

        window = min(self.smooth_window, T if T % 2 == 1 else T - 1)

        if window < 5:
            return seq

        out = seq.copy()

        for v in range(seq.shape[1]):
            for c in range(seq.shape[2]):
                out[:, v, c] = savgol_filter(
                    seq[:, v, c],
                    window_length=window,
                    polyorder=2,
                    mode="interp"
                )

        return out.astype(np.float32)

    # @staticmethod
    # def normalize_scale(seq):
    #     """
    #     Normalize by median shoulder width.
    #     Use only if your training preprocessing had body scale normalization.
    #     """

    #     left_shoulder = seq[:, 4]
    #     right_shoulder = seq[:, 8]
    #     shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1)

    #     valid = shoulder_width > 1e-6

    #     if not np.any(valid):
    #         return seq

    #     scale = np.median(shoulder_width[valid])
    #     scale = max(scale, 1e-6)

    #     return seq / scale


    @staticmethod
    def normalize_scale(seq):
        """
        Normalize scale to match common NTU training preprocessing.
        """
        seq = seq.copy()
        
        max_abs = np.max(np.abs(seq))
        if max_abs > 0:
            seq = seq / max_abs * 1.5          
        
        return seq


class MMPoseWholeBodyExtractor:
    def __init__(self, pose2d="wholebody", device="cuda:0"):
        """
        You can use:
            pose2d="wholebody"

        Or pass an explicit config path:
            pose2d="configs/wholebody_2d_keypoint/..."
        """
        from mmpose.apis import MMPoseInferencer

        self.inferencer = MMPoseInferencer(
            pose2d=pose2d,
            device=device,
        )

    #@time_logger
    def extract(self, frame_bgr):
        """
        Args:
            frame_bgr: OpenCV BGR image.

        Returns:
            kpts: [K, 3] keypoints for selected person, or None.
        """

        result_generator = self.inferencer(
            frame_bgr,
            show=False,
            return_vis=False,
        )

        result = next(result_generator)

        persons = self._get_person_predictions(result)

        if len(persons) == 0:
            return None

        person = self._select_main_person(persons)

        kpts = self._person_to_kpts(person)

        return kpts

    @staticmethod
    def _get_person_predictions(result):
        preds = result.get("predictions", [])

        # MMPose often gives predictions as [image][person].
        if isinstance(preds, list) and len(preds) > 0:
            if isinstance(preds[0], list):
                return preds[0]
            return preds

        return []

    @staticmethod
    def _person_to_kpts(person):
        keypoints = person.get("keypoints", None)

        scores = (
            person.get("keypoint_scores", None)
            or person.get("keypoint_score", None)
            or person.get("scores", None)
        )

        if keypoints is None:
            return None

        keypoints = np.asarray(keypoints, dtype=np.float32)

        if scores is None:
            scores = np.ones((keypoints.shape[0],), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32)

        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            return None

        if scores.ndim != 1:
            scores = scores.reshape(-1)

        K = min(keypoints.shape[0], scores.shape[0])

        out = np.zeros((K, 3), dtype=np.float32)
        out[:, :2] = keypoints[:K, :2]
        out[:, 2] = scores[:K]

        return out

    @staticmethod
    def _select_main_person(persons):
        """
        Select largest visible person.
        For one-person action recognition, this is usually enough.
        """

        best_person = persons[0]
        best_area = -1.0

        for p in persons:
            kpts = MMPoseWholeBodyExtractor._person_to_kpts(p)

            if kpts is None or len(kpts) == 0:
                continue

            valid = kpts[:, 2] > 0.2

            if np.sum(valid) < 5:
                continue

            xy = kpts[valid, :2]
            x1, y1 = np.min(xy, axis=0)
            x2, y2 = np.max(xy, axis=0)

            area = float((x2 - x1) * (y2 - y1))

            if area > best_area:
                best_area = area
                best_person = p

        return best_person


class VLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://integrate.api.nvidia.com/v1/chat/completions" 
        self.queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = True
        
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _encode_cv2_image(self, image: np.ndarray) -> str:
        """Convert cv2 image (BGR) to base64 JPEG"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        success, encoded = cv2.imencode('.jpg', rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise ValueError("Failed to encode image")
        
        return base64.b64encode(encoded.tobytes()).decode('utf-8')

    def _worker(self):
        """Background thread for API calls"""
        while self.running:
            try:
                task = self.queue.get(timeout=0.5)
                if task is None:
                    break
                    
                image, prompt, callback = task
                try:
                    base64_image = self._encode_cv2_image(image)
                    
                    payload = {
                        "model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", #cheap model
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "seed": 0,
                        "stream": False,
                        "top_p": 1
                    }

                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json"
                    }

                    response = requests.post(self.url, json=payload, headers=headers, stream=False, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        callback(result)
                        self.response_queue.put(result)
                    else:
                        print(f"Error: {response.status_code} - {response.text}")
                        callback({"error": response.text})
                except Exception as e:
                    print(f"Processing error: {e}")
                    if callback:
                        callback({"error": str(e)})
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue 
            except Exception as e:
                print(f"Worker thread error: {e}")

    def send_image(self, image: np.ndarray, prompt: str, callback=None):
        """Send image from main thread to worker"""
        if callback is None:
            callback = lambda x: print("Response:", x)
        
        self.queue.put((image, prompt, callback))

    def stop(self):
        self.running = False
        self.queue.put(None)
        self.worker.join()


def draw_ntu2d(frame, ntu2d, score, conf_thr=0.2):
    bones = [
        (0, 1), (1, 20), (20, 2), (2, 3),

        (20, 4), (4, 5), (5, 6), (6, 7),
        (7, 21), (6, 22),

        (20, 8), (8, 9), (9, 10), (10, 11),
        (11, 23), (10, 24),

        (0, 12), (12, 13), (13, 14), (14, 15),
        (0, 16), (16, 17), (17, 18), (18, 19),
    ]

    out = frame.copy()

    for a, b in bones:
        if score[a] >= conf_thr and score[b] >= conf_thr:
            pa = tuple(np.round(ntu2d[a]).astype(int))
            pb = tuple(np.round(ntu2d[b]).astype(int))
            cv2.line(out, pa, pb, (0, 255, 0), 2)

    for j in range(25):
        if score[j] >= conf_thr:
            p = tuple(np.round(ntu2d[j]).astype(int))
            cv2.circle(out, p, 4, (0, 0, 255), -1)
            cv2.putText(
                out,
                str(j),
                (p[0] + 3, p[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return out


def model_inference(model_input, model):
    model_tensor_input = torch.from_numpy(model_input).float().cuda()
    zeros = torch.zeros_like(model_tensor_input)
    input_tensor = torch.cat([model_tensor_input, zeros], dim=-1)
    detected_actions = []
    with torch.no_grad():
        output = model(input_tensor)
        output = output / output.max() # Normalize to [0, 1]
        #pred = torch.argmax(output, dim=1).item()
        values, indices = torch.topk(output[0], k=5)
        print(' '*12, 'MODEL OUTPUT',' '*12)
        print(len(indices))
        for i in range(len(indices)):
            try:
                if i == 0:
                    print(f"Action: {labels[indices[i]]}, Baseline confidence")
                else:
                    print(f"Action: {labels[indices[i]]}, Relative Confidence to top action: {values[i].item():.4f}")
                detected_actions.append(indices[i].item())
            except IndexError:
                pass
        print(' '*12, 'MODEL OUTPUT',' '*12)

    return detected_actions


# warmup the model with dummy input
def warmup(model):
    for _ in range(5):
        input_tensor = torch.randn(1, 3, 120, 25, 2).float().cuda()
        with torch.no_grad():
            output = model(input_tensor)


def handle_response(result):
    if "choices" in result:
        print(' '*25)
        print(' '*12, 'VLM RESPONSE',' '*12)
        print(result["choices"][0]["message"]["content"])
        print(' '*12, 'VLM RESPONSE',' '*12)
        print(' '*25)
    else:
        print("Raw response:", result)


def main():
    device = 'cuda:0'
    pose2d = 'wholebody'
    width = 1280
    height = 720
    body_height = 1.75
    conf_thr = 0.2
    target_frames = 60
    buffer_size = 40
    root_center = True
    root_joint = 20
    scale_normalize = True
    flip_y = False
    show = True

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Camera failed!!!")

    H, W = frame.shape[:2]

    fx = 1.2 * W
    fy = 1.2 * W
    cx = W / 2.0
    cy = H / 2.0

    intrinsics = CameraIntrinsics(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    print("Camera intrinsics:")
    print(intrinsics)

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    print("Loading MMPose")
    pose_extractor = MMPoseWholeBodyExtractor(
        pose2d=pose2d,
        device=device,
    )

    mapper = WholeBodyToNTU25Mapper(
        conf_thr=conf_thr,
    )

    fitter = NTU25Pseudo3DFitter(
        intrinsics=intrinsics,
        body_height_m=body_height,
        conf_thr=conf_thr,
        flip_y=flip_y,
    )

    preprocessor = NTUModelPreprocessor(
        target_frames=target_frames,
        root_center=root_center,
        root_joint=root_joint,
        scale_normalize=scale_normalize,
    )

    print('Loading Model')
    action_model = GraphModel(dim_in=3, dim=80, class_num=60).cuda()
    action_model.load_state_dict(torch.load('res/s_joint_mod.pt', map_location='cuda'))
    action_model.eval()

    xyz_buffer = deque(maxlen=buffer_size)

    pending_frame = frame

    frame_idx = 0
    last_time = time.time()

    warmup(action_model)
    clinet = VLMClient(api_key=NV_API_KEY)

    while True:
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1

        wholebody_kpts = pose_extractor.extract(frame)

        if wholebody_kpts is None:
            print("No person found.")
            if show:
                cv2.imshow("Camera Stream", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            continue

        ntu2d, score = mapper(wholebody_kpts)

        #print(ntu2d)
        ntu3d = fitter.lift(ntu2d, score)   # [25, 3]
        #print(ntu3d)
        xyz_buffer.append(ntu3d)

        if frame_idx % buffer_size == 0:
            seq_xyz = np.stack(list(xyz_buffer), axis=0)  # [T_raw, 25, 3]

            model_input, clean_seq = preprocessor(seq_xyz)

            #print(model_input)

            pred = model_inference(model_input, action_model)
            action_to_consider = next((x for x in pred if x in TARGET_IDS), None)
            if action_to_consider is not None:
                frame_to_show = cv2.resize(frame, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
                cv2.imshow('Detected medical condition!!!', frame_to_show)
                print(f"Possible medical condition detected! Detected possible action: {labels[action_to_consider]}")
                clinet.send_image(frame, PROMPT, callback=handle_response)
                cv2.waitKey(0)

        # if frame_idx % 10 == 0:
        #     now = time.time()
        #     fps = 10.0 / max(now - last_time, 1e-6)
        #     last_time = now

        #     print(
        #         f"[INFO] frame={frame_idx}, "
        #         f"buffer={len(xyz_buffer)}, "
        #         f"fps≈{fps:.2f}, "
        #     )

        if show:
            vis = draw_ntu2d(frame, ntu2d, score, conf_thr)

            cv2.putText(
                vis,
                f"Buffer {len(xyz_buffer)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            vis = cv2.resize(vis, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Camera Stream", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    gc.collect() #clean because no RAM
    torch.cuda.empty_cache()
    main()
