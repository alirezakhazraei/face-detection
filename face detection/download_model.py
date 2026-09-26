"""
Downloads the pretrained face-detection DNN model files from OpenCV's
official repositories.

The model is a Caffe-based SSD (Single Shot Detector) with a ResNet-10
backbone, trained by the OpenCV team specifically for face detection.
It's the standard "deep learning face detector" used in most OpenCV
tutorials and projects - much more accurate than Haar Cascades,
especially with angled faces, partial occlusion, or varied lighting.

Run this once before using the face detector:
    python download_model.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
)
CAFFEMODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)

PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
CAFFEMODEL_PATH = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


def download(url: str, destination: str) -> None:
    if os.path.exists(destination):
        print(f"Already exists, skipping: {destination}")
        return

    print(f"Downloading {os.path.basename(destination)} ...")
    urllib.request.urlretrieve(url, destination)
    print(f"Saved to {destination}")


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    download(PROTOTXT_URL, PROTOTXT_PATH)
    download(CAFFEMODEL_URL, CAFFEMODEL_PATH)
    print("\nDone. You can now run: python src/webcam_face_detection.py")


if __name__ == "__main__":
    main()
