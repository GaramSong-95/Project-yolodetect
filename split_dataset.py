#!/usr/bin/env python3
import os
import shutil
import random
import argparse

def split_dataset(dataset_dir, output_dir, train_ratio=0.8, seed=42):
    """
    dataset_dir 폴더 안의 .jpg/.png 이미지와 같은 이름의 .txt 파일을
    train/val로 분할해 output_dir 아래에 복사합니다.

    :param dataset_dir: 원본 이미지 + txt 이 섞여 있는 폴더
    :param output_dir: images/, labels/ 하위 train/val 폴더를 생성할 루트
    :param train_ratio: 학습셋 비율 (0.0~1.0)
    :param seed: 랜덤 셔플 시드 (재현성)
    """
    # 1) 이미지 파일 목록 수집
    exts = ('.jpg', '.jpeg', '.png')
    images = [f for f in os.listdir(dataset_dir) if f.lower().endswith(exts)]
    images.sort()
    random.seed(seed)
    random.shuffle(images)

    # 2) train/val 분할
    n_train = int(len(images) * train_ratio)
    train_imgs = images[:n_train]
    val_imgs   = images[n_train:]

    # 3) 출력 디렉토리 구조 생성
    for split in ('train', 'val'):
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # 4) 파일 복사 (.jpg/.png + .json)
    def copy_split(img_list, split):
        for img in img_list:
            # 이미지 복사
            src_img = os.path.join(dataset_dir, img)
            dst_img = os.path.join(output_dir, 'images', split, img)
            shutil.copy2(src_img, dst_img)

            # 라벨(txt) 복사
            label_name = os.path.splitext(img)[0] + '.txt'
            src_lbl = os.path.join(dataset_dir, label_name)
            dst_lbl = os.path.join(output_dir, 'labels', split, label_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, dst_lbl)
            else:
                print(f"[Warning] txt not found for {img}")

    copy_split(train_imgs, 'train')
    copy_split(val_imgs,   'val')

    print(f"Finished splitting: {len(train_imgs)} train, {len(val_imgs)} val images.")

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description="Split LabelMe dataset into train/val for YOLO v11"
    )
    p.add_argument('--dataset-dir', type=str, required=True,
                   help='원본 이미지+txt 폴더 경로')
    p.add_argument('--output-dir', type=str, default='.',
                   help='images/, labels/ 폴더가 생성될 루트')
    p.add_argument('--train-ratio', type=float, default=0.8,
                   help='학습셋 비율 (기본 0.8)')
    p.add_argument('--seed', type=int, default=42,
                   help='랜덤 시드 (기본 42)')
    args = p.parse_args()

    split_dataset(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )

