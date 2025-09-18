import time
import numpy as np
import cv2
import aidlite
import argparse

OBJ_CLASS_NUM = 80
NMS_THRESH = 0.45
BOX_THRESH = 0.5
MODEL_SIZE = 640

OBJ_NUMB_MAX_SIZE = 64
PROP_BOX_SIZE = (5 + OBJ_CLASS_NUM)
STRIDE8_SIZE = (MODEL_SIZE / 8)
STRIDE16_SIZE = (MODEL_SIZE / 16)
STRIDE32_SIZE = (MODEL_SIZE / 32)

anchors = [[10, 13, 16, 30, 33, 23],
           [30, 61, 62, 45, 59, 119],
           [116, 90, 156, 198, 373, 326]]

coco_class = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def eqprocess(image, size1, size2):
    h, w, _ = image.shape
    mask = np.zeros((size1, size2, 3), dtype=np.float32)
    scale1 = h / size1
    scale2 = w / size2
    if scale1 > scale2:
        scale = scale1
    else:
        scale = scale2
    img = cv2.resize(image, (int(w / scale), int(h / scale)))
    mask[:int(h / scale), :int(w / scale), :] = img
    return mask, scale


def xywh2xyxy(x):
    '''
    Box (center x, center y, width, height) to (x1, y1, x2, y2)
    '''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(box):
    '''
    Box (left_top x, left_top y, right_bottom x, right_bottom y) to (left_top x, left_top y, width, height)
    '''
    box[:, 2:] = box[:, 2:] - box[:, :2]
    return box


def NMS(dets, scores, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clip(0, img_shape[1], out=boxes[:, 0])  # x1
    boxes[:, 1].clip(0, img_shape[0], out=boxes[:, 1])  # y1
    boxes[:, 2].clip(0, img_shape[1], out=boxes[:, 2])  # x2
    boxes[:, 3].clip(0, img_shape[0], out=boxes[:, 3])  # y2


def detect_postprocess(prediction, img0shape, img1shape, conf_thres=0.25, iou_thres=0.45):
    '''
    检测输出后处理
    prediction: aidlite模型预测输出
    img0shape: 原始图片shape
    img1shape: 输入图片shape
    conf_thres: 置信度阈值
    iou_thres: IOU阈值
    return: list[np.ndarray(N, 5)], 对应类别的坐标框信息, xywh、conf
    '''
    h, w, _ = img1shape
    valid_condidates = prediction[prediction[..., 4] > conf_thres]
    valid_condidates[:, 5:] *= valid_condidates[:, 4:5]
    valid_condidates[:, :4] = xywh2xyxy(valid_condidates[:, :4])

    max_det = 300
    max_wh = 7680
    max_nms = 30000
    valid_condidates[:, 4] = valid_condidates[:, 5:].max(1)
    valid_condidates[:, 5] = valid_condidates[:, 5:].argmax(1)
    sort_id = np.argsort(valid_condidates[:, 4])[::-1]
    valid_condidates = valid_condidates[sort_id[:max_nms]]
    boxes, scores = valid_condidates[:, :4] + valid_condidates[:, 5:6] * max_wh, valid_condidates[:, 4]
    index = NMS(boxes, scores, iou_thres)[:max_det]
    out_boxes = valid_condidates[index]
    clip_coords(out_boxes[:, :4], img0shape)
    out_boxes[:, :4] = xyxy2xywh(out_boxes[:, :4])
    print("检测到{}个区域".format(len(out_boxes)))
    return out_boxes


def draw_detect_res(img, det_pred):
    '''
    检测结果绘制
    '''
    img = img.astype(np.uint8)
    color_step = int(255 / len(coco_class))
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])

        print(i + 1, [x1, y1, x2, y2], score, coco_class[cls_id])

        cv2.putText(img, f'{coco_class[cls_id]}', (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.rectangle(img, (x1, y1), (x2 + x1, y2 + y1), (0, int(cls_id * color_step), int(255 - cls_id * color_step)),
                      thickness=2)

    return img


class Detect():
    # YOLOv5 Detect head for detection models
    def __init__(self, nc=80, anchors=(), stride=[], image_size=640):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.stride = stride
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid, self.anchor_grid = [0] * self.nl, [0] * self.nl
        self.anchors = np.array(anchors, dtype=np.float32).reshape(self.nl, -1, 2)

        base_scale = image_size // 8
        for i in range(self.nl):
            self.grid[i], self.anchor_grid[i] = self._make_grid(base_scale // (2 ** i), base_scale // (2 ** i), i)

    def _make_grid(self, nx=20, ny=20, i=0):
        y, x = np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32)
        yv, xv = np.meshgrid(y, x)
        yv, xv = yv.T, xv.T
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid = np.stack((xv, yv), 2)
        grid = grid[np.newaxis, np.newaxis, ...]
        grid = np.repeat(grid, self.na, axis=1) - 0.5
        anchor_grid = self.anchors[i].reshape((1, self.na, 1, 1, 2))
        anchor_grid = np.repeat(anchor_grid, repeats=ny, axis=2)
        anchor_grid = np.repeat(anchor_grid, repeats=nx, axis=3)
        return grid, anchor_grid

    def sigmoid(self, arr):
        return 1 / (1 + np.exp(-arr))

    def __call__(self, x):
        z = []  # inference output
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].reshape(bs, self.na, self.no, ny, nx).transpose(0, 1, 3, 4, 2)
            y = self.sigmoid(x[i])
            y[..., 0:2] = (y[..., 0:2] * 2. + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            z.append(y.reshape(bs, self.na * nx * ny, self.no))

        return np.concatenate(z, 1)

def main():
    args = parser_args()
    target_model = args.target_model
    model_type = args.model_type
    size = int(args.size)
    imgs = args.imgs
    invoke_nums = int(args.invoke_nums)
    print("Start main ... ...")
    # aidlite.set_log_level(aidlite.LogLevel.INFO)
    # aidlite.log_to_stderr()
    # print(f"Aidlite library version : {aidlite.get_library_version()}")
    # print(f"Aidlite python library version : {aidlite.get_py_library_version()}")

    config = aidlite.Config.create_instance()
    if config is None:
        print("Create config failed !")
        return False
    
    
    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    if model_type.lower()=="qnn":
        config.framework_type = aidlite.FrameworkType.TYPE_QNN
    elif model_type.lower()=="snpe2" or model_type.lower()=="snpe":
        config.framework_type = aidlite.FrameworkType.TYPE_SNPE2
        
    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.is_quantify_model = 1

    
    model = aidlite.Model.create_instance(target_model)
    if model is None:
        print("Create model failed !")
        return False
    input_shapes = [[1, size, size, 3]]
    output_shapes = [[1, 20, 20, 255], [1, 40, 40, 255], [1, 80, 80, 255]]
    model.set_model_properties(input_shapes, aidlite.DataType.TYPE_FLOAT32,
                               output_shapes, aidlite.DataType.TYPE_FLOAT32)

    interpreter = aidlite.InterpreterBuilder.build_interpretper_from_model_and_config(model, config)
    if interpreter is None:
        print("build_interpretper_from_model_and_config failed !")
        return None
    result = interpreter.init()
    if result != 0:
        print(f"interpreter init failed !")
        return False
    result = interpreter.load_model()
    if result != 0:
        print("interpreter load model failed !")
        return False
    print("detect model load success!")
    
    # image process
    frame = cv2.imread(imgs)
    # 图片做等比缩放
    img_processed = np.copy(frame)
    [height, width, _] = img_processed.shape
    length = max((height, width))
    scale = length / size
    ratio=[scale,scale]
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = img_processed
    img_input = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_input=cv2.resize(img_input,(size,size))
    
    mean_data=[0, 0, 0]
    std_data=[255, 255, 255]
    img_input = (img_input-mean_data)/std_data  # HWC

    img_input = img_input.astype(np.float32)
    
    
    # qnn run
    invoke_time=[]
    for i in range(invoke_nums):
        result = interpreter.set_input_tensor(0, img_input.data)
        if result != 0:
            print("interpreter set_input_tensor() failed")
        
        t1=time.time()
        result = interpreter.invoke()
        cost_time = (time.time()-t1)*1000
        invoke_time.append(cost_time)
        
        if result != 0:
            print("interpreter set_input_tensor() failed")
    
    out1 = interpreter.get_output_tensor(0)
    out2 = interpreter.get_output_tensor(1)
    out3 = interpreter.get_output_tensor(2)
    
    output  = [out1,out2,out3]
    output = sorted(output,key=len)
   

    result = interpreter.destory()
    
    ## time 统计
    max_invoke_time = max(invoke_time)
    min_invoke_time = min(invoke_time)
    mean_invoke_time = sum(invoke_time)/invoke_nums
    var_invoketime=np.var(invoke_time)
    print("=======================================")
    print(f"QNN inference {invoke_nums} times :\n --mean_invoke_time is {mean_invoke_time} \n --max_invoke_time is {max_invoke_time} \n --min_invoke_time is {min_invoke_time} \n --var_invoketime is {var_invoketime}")
    print("=======================================")
        
    ##  后处理
    stride = [8, 16, 32]
    yolo_head = Detect(OBJ_CLASS_NUM, anchors, stride, MODEL_SIZE)
    validCount0 = output[2].reshape(*output_shapes[2]).transpose(0, 3, 1, 2)
    validCount1 = output[1].reshape(*output_shapes[1]).transpose(0, 3, 1, 2)
    validCount2 = output[0].reshape(*output_shapes[0]).transpose(0, 3, 1, 2)
    pred = yolo_head([validCount0, validCount1, validCount2])
    det_pred = detect_postprocess(pred, frame.shape, [MODEL_SIZE, MODEL_SIZE, 3], conf_thres=0.5, iou_thres=0.45)
    det_pred[np.isnan(det_pred)] = 0.0
    det_pred[:, :4] = det_pred[:, :4] * scale
    res_img = draw_detect_res(frame, det_pred)
    save_path="./python/yolov5s_result.jpg"
    cv2.imwrite(save_path, res_img)   
    print("图片保存在",save_path)
    print("=======================================")

    return True

    
def parser_args():
    parser = argparse.ArgumentParser(description="Run model benchmarks")
    parser.add_argument('--target_model',type=str,default='./models/cutoff_yolov5s_qcs8550_w8a8.qnn231.ctx.bin',help="inference model path")
    parser.add_argument('--imgs',type=str,default='./python/bus.jpg',help="Predict images path")
    parser.add_argument('--invoke_nums',type=str,default=10,help="Inference nums")
    parser.add_argument('--model_type',type=str,default='QNN',help="run backend")
    parser.add_argument('--size',type=str,default=640,help="model input size")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()

