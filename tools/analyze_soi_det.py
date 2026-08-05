#!/usr/bin/env python
import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


CLASSES = (
    'ship', 'aircraft', 'car', 'tank', 'bridge', 'harbor',
    'small-vehicle', 'large-vehicle', 'plane', 'Ship', 'Harbor',
    'tennis-court', 'soccer-ball-field', 'ground-track-field',
    'baseball-diamond', 'swimming-pool', 'roundabout', 'basketball-court',
    'storage-tank', 'Bridge', 'helicopter', 'CAR', 'BUS', 'FERIGHT_CAR',
    'TRUCK', 'VAN'
)

GROUPS = {
    'ship': ['ship', 'Ship'],
    'aircraft_plane': ['aircraft', 'plane'],
    'car_vehicle': ['car', 'CAR', 'small-vehicle', 'large-vehicle', 'BUS',
                    'FERIGHT_CAR', 'TRUCK', 'VAN'],
    'bridge': ['bridge', 'Bridge'],
    'harbor': ['harbor', 'Harbor'],
    'tank': ['tank', 'storage-tank'],
}

PALETTE = [
    (220, 120, 60), (220, 220, 60), (220, 20, 120), (220, 20, 220),
    (220, 20, 0), (220, 120, 0), (220, 20, 60), (119, 11, 32),
    (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100),
    (0, 80, 100), (0, 0, 192), (250, 170, 30), (100, 170, 30),
    (220, 220, 0), (175, 116, 175), (250, 0, 30), (165, 42, 42),
    (0, 226, 252), (255, 128, 0), (255, 0, 255), (0, 255, 255),
    (255, 193, 193), (0, 51, 153)
]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def coco_images_and_anns(json_path, img_dir):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    categories = {c['id']: c['name'] for c in data.get('categories', [])}
    images = {img['id']: img for img in data.get('images', [])}
    anns_by_img = defaultdict(list)
    for ann in data.get('annotations', []):
        anns_by_img[ann['image_id']].append(ann)

    items = []
    for img_id, info in images.items():
        file_name = info.get('file_name') or info.get('filename')
        path = Path(img_dir) / file_name
        anns = []
        for ann in anns_by_img.get(img_id, []):
            cls = categories.get(ann.get('category_id'), str(ann.get('category_id')))
            if 'bbox' not in ann:
                continue
            x, y, w, h = ann['bbox'][:4]
            anns.append(dict(cls=cls, poly=None, bbox=[x, y, w, h], area=float(w * h)))
        items.append(dict(path=path, anns=anns))
    return items


def dota_images_and_anns(ann_dir, img_dir):
    items = []
    for ann_path in sorted(Path(ann_dir).glob('*.txt')):
        stem = ann_path.stem
        img_path = None
        for ext in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
            cand = Path(img_dir) / f'{stem}{ext}'
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            img_path = Path(img_dir) / f'{stem}.png'

        anns = []
        with open(ann_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                poly = np.array([float(x) for x in parts[:8]], dtype=np.float32)
                cls = parts[8]
                xs = poly[0::2]
                ys = poly[1::2]
                x, y = float(xs.min()), float(ys.min())
                w, h = float(xs.max() - xs.min()), float(ys.max() - ys.min())
                area = float(abs(cv2.contourArea(poly.reshape(-1, 2))))
                anns.append(dict(cls=cls, poly=poly.tolist(), bbox=[x, y, w, h], area=area))
        items.append(dict(path=img_path, anns=anns))
    return items


def size_bucket(area_ratio):
    if area_ratio < 0.01:
        return 'small(<1%)'
    if area_ratio < 0.05:
        return 'medium(1%-5%)'
    return 'large(>=5%)'


def draw_sample(img, anns, out_path):
    vis = img.copy()
    for ann in anns:
        cls = ann['cls']
        color = PALETTE[CLASSES.index(cls) % len(PALETTE)] if cls in CLASSES else (0, 255, 255)
        if ann.get('poly') is not None:
            pts = np.array(ann['poly'], dtype=np.int32).reshape(-1, 2)
            cv2.polylines(vis, [pts], True, color, 2)
            x, y = pts[:, 0].min(), pts[:, 1].min()
        else:
            x, y, w, h = [int(v) for v in ann['bbox']]
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, cls, (int(x), max(12, int(y) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def analyze_dataset(name, items, out_dir, max_images=None, sample_per_dataset=24):
    rows = []
    class_counter = Counter()
    image_obj_counts = []
    res_counter = Counter()
    size_counter = Counter()
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    brightness_hist = np.zeros(256, dtype=np.int64)
    class_area_ratios = defaultdict(list)
    sample_candidates = []

    selected = items[:max_images] if max_images else items
    for idx, item in enumerate(selected):
        try:
            img = read_image(item['path'])
        except FileNotFoundError:
            print(f'[WARN] missing image: {item["path"]}')
            continue
        h, w = img.shape[:2]
        image_area = max(1, h * w)
        res_counter[f'{w}x{h}'] += 1
        image_obj_counts.append(len(item['anns']))

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
        pixel_sum += rgb.reshape(-1, 3).sum(axis=0)
        pixel_sq_sum += (rgb.reshape(-1, 3) ** 2).sum(axis=0)
        pixel_count += h * w
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness_hist += np.bincount(gray.reshape(-1), minlength=256)

        for ann in item['anns']:
            cls = ann['cls']
            class_counter[cls] += 1
            area_ratio = float(ann['area']) / image_area
            bucket = size_bucket(area_ratio)
            size_counter[bucket] += 1
            class_area_ratios[cls].append(area_ratio)
            rows.append({
                'dataset': name,
                'image': str(item['path']),
                'width': w,
                'height': h,
                'num_objects_in_image': len(item['anns']),
                'class': cls,
                'bbox_area_ratio': area_ratio,
                'size_bucket': bucket,
            })

        if item['anns']:
            sample_candidates.append((len(item['anns']), idx, item, img))

    mean = pixel_sum / max(1, pixel_count)
    std = np.sqrt(pixel_sq_sum / max(1, pixel_count) - mean ** 2)
    obj_counts = np.array(image_obj_counts or [0])

    summary = {
        'dataset': name,
        'num_images': len(image_obj_counts),
        'num_objects': int(sum(image_obj_counts)),
        'objects_per_image': {
            'mean': float(obj_counts.mean()),
            'median': float(np.median(obj_counts)),
            'min': int(obj_counts.min()),
            'max': int(obj_counts.max()),
        },
        'top_resolutions': res_counter.most_common(20),
        'pixel_rgb_mean': mean.round(3).tolist(),
        'pixel_rgb_std': std.round(3).tolist(),
        'class_counts': dict(class_counter),
        'size_buckets': dict(size_counter),
        'class_area_ratio_mean': {
            k: float(np.mean(v)) for k, v in class_area_ratios.items()
        },
    }

    sample_dir = Path(out_dir) / 'visual_samples' / name
    ensure_dir(sample_dir)
    sample_candidates.sort(reverse=True)
    step = max(1, len(sample_candidates) // max(1, sample_per_dataset))
    for n, (_, _, item, img) in enumerate(sample_candidates[::step][:sample_per_dataset]):
        draw_sample(img, item['anns'], sample_dir / f'{n:03d}_{Path(item["path"]).stem}.jpg')

    hist_path = Path(out_dir) / f'{name}_brightness_hist.csv'
    with open(hist_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['gray_value', 'pixel_count'])
        for i, v in enumerate(brightness_hist):
            writer.writerow([i, int(v)])

    return summary, rows


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='/mnt/data/jfu/workspace/SM3Det-main/data/SOI_Det')
    parser.add_argument('--out-dir', default='work_dirs/dataset_analysis')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--sample-per-dataset', type=int, default=24)
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    datasets = {
        'sar': coco_images_and_anns(
            root / 'SARDet_50K/Annotations/after_merge_train.json',
            root / 'SARDet_50K/JPEGImages'),
        'rgb': dota_images_and_anns(
            root / 'DOTA_800pix/train/annfiles',
            root / 'DOTA_800pix/train/images'),
        'ifr': dota_images_and_anns(
            root / 'DroneVehicle/dota_train/annfiles',
            root / 'DroneVehicle/dota_train/png_images'),
    }

    all_rows = []
    summaries = {}
    for name, items in datasets.items():
        print(f'Analyzing {name}: {len(items)} images')
        summary, rows = analyze_dataset(
            name, items, out_dir, args.max_images, args.sample_per_dataset)
        summaries[name] = summary
        all_rows.extend(rows)

    class_table = []
    for i, cls in enumerate(CLASSES):
        class_table.append({
            'global_id': i,
            'class': cls,
            'sar_count': summaries['sar']['class_counts'].get(cls, 0),
            'rgb_count': summaries['rgb']['class_counts'].get(cls, 0),
            'ifr_count': summaries['ifr']['class_counts'].get(cls, 0),
        })

    group_rows = []
    for group, members in GROUPS.items():
        row = {'semantic_group': group, 'members': ','.join(members)}
        for ds in ('sar', 'rgb', 'ifr'):
            row[f'{ds}_count'] = sum(summaries[ds]['class_counts'].get(m, 0) for m in members)
        group_rows.append(row)

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    write_csv(out_dir / 'objects.csv', all_rows)
    write_csv(out_dir / 'class_mapping_counts.csv', class_table)
    write_csv(out_dir / 'semantic_group_counts.csv', group_rows)

    print('\nSaved analysis to:', out_dir)
    print('Key files:')
    print('  summary.json')
    print('  objects.csv')
    print('  class_mapping_counts.csv')
    print('  semantic_group_counts.csv')
    print('  visual_samples/{sar,rgb,ifr}/*.jpg')


if __name__ == '__main__':
    main()
