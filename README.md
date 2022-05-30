# Eagle_Detection

## Inference and Instalation Process:-
### Yolov5:-
#### Installation:-
1. ```!git clone https://github.com/ultralytics/yolov5```
2. ```cd yolov5```
3. ```pip install -qr requirements.txt```

#### Inference:-
if ```yolo_inference.py``` not working than follow the following command to run yolo inference.
```python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source /content/yolov5/COCO-3/test/images```

### Retinanet:-
#### Installation:-
1. ```git clone https://github.com/fizyr/keras-retinanet.git```
2. ```cd keras-retinanet/```
3. ```pip install -r requirements.txt```
4. ```pip install -e .```
5. ```python setup.py build_ext --inplace```
6. ```pip install --user git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI```
7. Now you can change the path load the model from solution_document and start inferencing the result using ```retinanet_inference.py``` file.

