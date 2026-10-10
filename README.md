# Object Detection — Jupyter Project

Extract the ZIP, open Object_Detection.ipynb in Jupyter, run the installation cell, restart the kernel, and run the remaining cells in order.

Uses a pretrained SSDLite MobileNetV3 detector. Outputs bounding boxes, category labels, confidence scores, per-image counts and CSV exports. CPU supported; CUDA chosen when available. No API key.

The model weights and demo image are downloaded on your computer on first run. They are NOT bundled in this ZIP. Alternatively use your own JPG/PNG. There is no custom training dataset or training stage in this inference project. The pretrained detector uses COCO categories; it will not identify arbitrary custom parts or defects.

Internet required for initial dependency and weight downloads. For compatible installations see https://pytorch.org/get-started/locally/ .

Notebook and equivalent Python source are included. Code syntax was checked, but inference could not be run in the creation environment because PyTorch was unavailable.
