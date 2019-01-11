import sys
import os
import pathlib
import shutil
import cv2
import glob
import numpy as np
from PIL import Image

def load_name_images(image_path_pattern):
    name_images = []
    # 指定したパスパターンに一致するファイルの取得
    image_paths = glob.glob(image_path_pattern)
    # ファイルごとの読み込み
    for image_path in image_paths:
        path = pathlib.Path(image_path)
        # ファイルパス
        fullpath = str(path.resolve())
        print(f"画像ファイル（絶対パス）:{fullpath}")
        # ファイル名
        filename = path.name
        print(f"画像ファイル（名前）:{filename}")
        # 画像読み込み
        image = cv2.imread(fullpath)
        if image is None:
            print(f"画像ファイル[{fullpath}]を読み込めません")
            continue
        name_images.append((filename,image))
    return name_images

def detect_image_face(file_path, image, cascade_filepath):
    # 画像ファイルのグレースケール化
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # カスケードファイルの読み込み
    cascade = cv2.CascadeClassifier(cascade_filepath)
    # 顔認識
    faces = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))
    if len(faces) == 0:
        print(f"顔認識失敗")
        return
    # 1つ以上の顔を認識
    face_count = 1
    for rect in faces:
        x, y, width, height = rect
        face_image = image[y:y+height, x:x+width]
        if face_image.shape[0] > 64:
            face_image = cv2.resize(face_image, (64, 64))
        print(face_image.shape)
        # 保存
        path = pathlib.Path(file_path)
        directory = str(path.parent.resolve())
        filename = path.stem
        extension = path.suffix
        output_path = os.path.join(directory, f"{filename}_{face_count:03}{extension}")
        print(f"出力ファイル（絶対パス）:{output_path}")
        cv2.imwrite(output_path, face_image)
        face_count = face_count + 1

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Origin Image Pattern
IMAGE_PATH_PATTERN = "./origin_image/*"
# Output Directory
OUTPUT_IMAGE_DIR = "./face_image"
# OpenCV Cascades File
CASCADE_FILE_PATH = "C:\\Develop\\Bin\\opencv344\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml"

def main():
    print("===================================================================")
    print("イメージ顔認識 OpenCV 利用版")
    print("指定した画像ファイルの正面顔を認識してグレースケール化とサイズ変更64x64を行います。")
    print("===================================================================")

    # ディレクトリの作成
    if os.path.isdir(OUTPUT_IMAGE_DIR) == False:
        os.mkdir(OUTPUT_IMAGE_DIR)

    # 画像ファイルの読み込み
    name_images = load_name_images(IMAGE_PATH_PATTERN)

    # 画像ごとの顔認識
    for name_image in name_images:
        file_path = os.path.join(OUTPUT_IMAGE_DIR, f"{name_image[0]}")
        image = name_image[1]
        cascade_filepath = CASCADE_FILE_PATH
        detect_image_face(file_path, image, cascade_filepath)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()
