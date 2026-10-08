import os

# OpenCV reads this once, when cv2 is first imported, and from then on
# refuses to decode an image whose header declares more pixels. Without it a
# tiny, highly compressible file (a 2.5 KB WebP can declare 8000x8000) is
# fully decoded before splitter.MAX_MEGAPIXELS is checked. Setting it here
# runs before any app module imports cv2.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "40000000")
