entrypoint: python runway_model.py
python: 3.6
cuda: 9.0
spec:
  gpu: True
  cpu: True
build_steps:
  - pip install torch==1.1.0 runway-python numpy==1.16.4
