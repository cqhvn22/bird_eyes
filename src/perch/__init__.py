"""Trained NCNN stereo perch detection for camera and video inputs."""
import os

# Set before importing NumPy/OpenCV; NCNN manages its own inference threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

__version__ = "1.0.0"
