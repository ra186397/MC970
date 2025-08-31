import argparse
import glob
import os
from typing import List, Tuple

import cv2
import numpy as np


# ----------------------------
# 1) Utilidades de I/O
# ----------------------------
def load_images(img_dir_or_list: str) -> List[np.ndarray]:
    """Carrega imagens de um diretório (ordenadas por nome) ou de uma lista de caminhos separados por vírgula."""
    if os.path.isdir(img_dir_or_list):
        paths = sorted(glob.glob(os.path.join(img_dir_or_list, "*.*")))
    else:
        paths = [p.strip() for p in img_dir_or_list.split(",")]

    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Não consegui ler {p}. Pulando.")
            continue
        imgs.append(img)
    if len(imgs) < 2:
        raise ValueError("Preciso de pelo menos 2 imagens.")
    return imgs


def ensure_out(outdir: str):
    os.makedirs(outdir, exist_ok=True)


# ----------------------------
# 2) Detecção/descrição de features
# ----------------------------
def build_detector(method: str):
    method = method.lower()
    if method == "sift":
        try:
            return cv2.SIFT_create()
        except Exception:
            raise RuntimeError("SIFT requer opencv-contrib-python.")
    elif method == "orb":
        return cv2.ORB_create(nfeatures=4000)
    elif method == "akaze":
        return cv2.AKAZE_create()
    else:
        raise ValueError("Detector inválido. Use: sift, orb ou akaze.")


def detect_and_describe(img: np.ndarray, detector) -> Tuple[np.ndarray, np.ndarray]:
    kps = detector.detect(img, None)
    kps, desc = detector.compute(img, kps)
    return np.array(kps), desc


def draw_keypoints(img: np.ndarray, kps, out_path: str):
    vis = cv2.drawKeypoints(img, list(kps), None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_path, vis)


# ----------------------------
# 3) Matching + Lowe ratio
# ----------------------------
def build_matcher(detector_name: str, use_flann: bool):
    detector_name = detector_name.lower()
    if use_flann:
        if detector_name in ["sift", "akaze"]:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=200)
            return cv2.FlannBasedMatcher(index_params, search_params), True
        else:
            # ORB/BRIEF -> binário -> usa LSH no FLANN
            index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                table_number=12, key_size=20, multi_probe_level=2)
            search_params = dict(checks=200)
            return cv2.FlannBasedMatcher(index_params, search_params), False
    else:
        # BFMatcher com norma adequada
        if detector_name in ["sift", "akaze"]:
            return cv2.BFMatcher(cv2.NORM_L2), True
        else:
            return cv2.BFMatcher(cv2.NORM_HAMMING), False


def match_features(descA, descB, matcher, ratio=0.75):
    raw = matcher.knnMatch(descA, descB, k=2)
    good = []
    for m, n in raw:
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def draw_matches(imgA, kpsA, imgB, kpsB, matches, out_path: str, max_draw=80):
    matches_to_draw = sorted(matches, key=lambda m: m.distance)[:max_draw]
    vis = cv2.drawMatches(imgA, list(kpsA), imgB, list(kpsB), matches_to_draw, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, vis)


# ----------------------------
# 4) RANSAC Homografia
# ----------------------------
def estimate_homography(kpsA, kpsB, matches, ransac_thresh: float = 4.0):
    if len(matches) < 4:
        return None, None

    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransac_thresh)
    inliers = mask.ravel().tolist() if mask is not None else None
    return H, inliers


def draw_inlier_matches(imgA, kpsA, imgB, kpsB, matches, inliers, out_path: str):
    if inliers is None:
        return
    inlier_matches = [m for m, keep in zip(matches, inliers) if keep]
    draw_matches(imgA, kpsA, imgB, kpsB, inlier_matches, out_path)


# ----------------------------
# 5) Warp + Blending (feathering)
# ----------------------------
def warp_two_images(img1, img2, H):
    """
    Warpa img1 para o plano de img2 com homografia H (img1->img2) e retorna
    (canvas, offset_x, offset_y) onde canvas contém ambas as imagens posicionadas.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    warped_corners1 = cv2.perspectiveTransform(corners1, H)

    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    [xmin, ymin] = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    [xmax, ymax] = np.ceil(all_corners.max(axis=0).ravel()).astype(int)

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    canvas_w = xmax - xmin
    canvas_h = ymax - ymin

    warped_img1 = cv2.warpPerspective(img1, T @ H, (canvas_w, canvas_h))
    warped_img2 = np.zeros_like(warped_img1)
    warped_img2[ty:ty + h2, tx:tx + w2] = img2

    return warped_img1, warped_img2


def feather_blend(warpedA, warpedB, feather=50):
    """
    Faz um feathering linear simples na região de overlap.
    """
    grayA = cv2.cvtColor((warpedA > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor((warpedB > 0).astype(np.uint8) * 255, cv2.COLOR_BGR2GRAY)

    maskA = (grayA > 0).astype(np.uint8)
    maskB = (grayB > 0).astype(np.uint8)

    overlap = (maskA & maskB).astype(np.uint8)

    # distância até borda para pesos suaves
    distA = cv2.distanceTransform(maskA, cv2.DIST_L2, 5)
    distB = cv2.distanceTransform(maskB, cv2.DIST_L2, 5)

    # normaliza pesos no overlap
    eps = 1e-6
    wA = distA / (distA + distB + eps)
    wB = 1.0 - wA

    # suaviza pesos (feather)
    if feather > 0:
        k = max(1, int(feather // 2 * 2 + 1))
        wA = cv2.GaussianBlur(wA, (k, k), 0)
        wB = cv2.GaussianBlur(wB, (k, k), 0)

    wA = wA[..., None]
    wB = wB[..., None]

    out = np.zeros_like(warpedA, dtype=np.float32)
    out += warpedA.astype(np.float32) * (wA * maskA[..., None])
    out += warpedB.astype(np.float32) * (wB * maskB[..., None])

    # nas regiões exclusivas, usar a imagem correspondente
    onlyA = (maskA == 1) & (maskB == 0)
    onlyB = (maskB == 1) & (maskA == 0)
    out[onlyA] = warpedA[onlyA]
    out[onlyB] = warpedB[onlyB]

    return np.clip(out, 0, 255).astype(np.uint8)


# ----------------------------
# 6) Stitch progressivo (da esquerda pra direita)
# ----------------------------
def stitch_pair(img_left, img_right, detector_name, use_flann, ratio, ransac_thresh, outdir, step_tag):
    detector = build_detector(detector_name)

    kpsL, descL = detect_and_describe(img_left, detector)
    kpsR, descR = detect_and_describe(img_right, detector)

    draw_keypoints(img_left, kpsL, os.path.join(outdir, f"{step_tag}_kps_left.jpg"))
    draw_keypoints(img_right, kpsR, os.path.join(outdir, f"{step_tag}_kps_right.jpg"))

    matcher, _ = build_matcher(detector_name, use_flann)
    good = match_features(descL, descR, matcher, ratio)
    draw_matches(img_left, kpsL, img_right, kpsR, good, os.path.join(outdir, f"{step_tag}_matches.jpg"))

    H, inliers = estimate_homography(kpsL, kpsR, good, ransac_thresh)
    if H is None:
        raise RuntimeError("Homografia não encontrada (matches insuficientes).")
    draw_inlier_matches(img_left, kpsL, img_right, kpsR, good, inliers,
                        os.path.join(outdir, f"{step_tag}_inliers.jpg"))

    warpedL, warpedR = warp_two_images(img_left, img_right, H)
    cv2.imwrite(os.path.join(outdir, f"{step_tag}_warped_left_on_right.jpg"), warpedL)
    cv2.imwrite(os.path.join(outdir, f"{step_tag}_canvas_right.jpg"), warpedR)

    blended = feather_blend(warpedL, warpedR, feather=50)
    cv2.imwrite(os.path.join(outdir, f"{step_tag}_blended.jpg"), blended)

    return blended


def stitch_sequence(imgs: List[np.ndarray], detector_name="sift", use_flann=True,
                    ratio=0.75, ransac_thresh=4.0, outdir="outputs") -> np.ndarray:
    ensure_out(outdir)
    panorama = imgs[0]
    for i in range(1, len(imgs)):
        step_tag = f"step_{i:02d}"
        print(f"[INFO] Alinhando imagem {i} com {detector_name.upper()} ...")
        panorama = stitch_pair(panorama, imgs[i], detector_name, use_flann, ratio, ransac_thresh, outdir, step_tag)

    return panorama


# ----------------------------
# 7) CLI
# ----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Gerador de panorama (SIFT/ORB/AKAZE + RANSAC + feathering).")
    ap.add_argument("--images", required=True,
                    help="Diretório com imagens (ordenadas por nome) OU lista de caminhos separados por vírgula.")
    ap.add_argument("--detector", default="sift", choices=["sift", "orb", "akaze"],
                    help="Detector/descritor de features.")
    ap.add_argument("--no-flann", action="store_true", help="Usa BFMatcher em vez de FLANN.")
    ap.add_argument("--ratio", type=float, default=0.75, help="Parâmetro do Lowe ratio test.")
    ap.add_argument("--ransac", type=float, default=4.0, help="Threshold do RANSAC para homografia.")
    ap.add_argument("--outdir", default="outputs", help="Pasta de saídas.")
    ap.add_argument("--save_final", default="panorama.jpg", help="Nome do arquivo final (dentro de outdir).")
    return ap.parse_args()


def main():
    args = parse_args()
    imgs = load_images(args.images)

    # opcional: redimensionar levemente para acelerar (se quiser)
    # imgs = [cv2.resize(im, None, fx=0.7, fy=0.7) for im in imgs]

    pano = stitch_sequence(
        imgs,
        detector_name=args.detector,
        use_flann=(not args.no_flann),
        ratio=args.ratio,
        ransac_thresh=args.ransac,
        outdir=args.outdir
    )
    cv2.imwrite(os.path.join(args.outdir, args.save_final), pano)
    print(f"[OK] Panorama salvo em {os.path.join(args.outdir, args.save_final)}")


if __name__ == "__main__":
    main()
