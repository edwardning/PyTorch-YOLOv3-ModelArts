# -*- coding: utf-8 -*-
import json
import codecs
from collections import OrderedDict
from models import *
from my_utils.utils import *
from my_utils.datasets import *


from model_service.pytorch_model_service import PTServingBaseService

import time
from metric.metrics_manager import MetricsManager
import log
logger = log.getLogger(__name__)


class ObjectDetectionService(PTServingBaseService):
    def __init__(self, model_name, model_path):
        # make sure these files exist
        self.model_name = model_name
        self.model_path = os.path.join(os.path.dirname(__file__), 'models_best.pth')
        self.classes_path = os.path.join(os.path.dirname(__file__), 'train_classes.txt')
        self.model_def = os.path.join(os.path.dirname(__file__), 'yolov3-44.cfg')
        self.label_map = parse_classify_rule(os.path.join(os.path.dirname(__file__), 'classify_rule.json'))

        self.input_image_key = 'images'
        self.score = 0.3
        self.iou = 0.45
        self.img_size = 416
        self.classes = self._get_class()
        # define and load YOLOv3 model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Darknet(self.model_def, img_size=self.img_size).to(device)
        if self.model_path.endswith(".weights"):
            # Load darknet weights
            self.model.load_darknet_weights(self.model_path)
        else:
            # Load checkpoint weights
            self.model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        print('load weights file success')
        self.model.eval()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with codecs.open(classes_path, 'r', 'utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _preprocess(self, data):
        preprocessed_data = {}
        for k, v in data.items():
            for file_name, file_content in v.items():
                img = Image.open(file_content)
                # store image size (height, width)
                shape = (img.size[1], img.size[0])
                # convert to tensor
                img = transforms.ToTensor()(img)
                # Pad to square resolution
                img, _ = pad_to_square(img, 0)
                # Resize
                img = resize(img, 416)
                # unsqueeze
                img = img.unsqueeze(0)

                preprocessed_data[k] = [img, shape]
        return preprocessed_data

    def _inference(self, data):
        """
        model inference function
        Here are a inference example of resnet, if you use another model, please modify this function
        """
        img, shape = data[self.input_image_key]

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        input_imgs = Variable(img.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = self.model(input_imgs)
            detections = non_max_suppression(detections, self.score, self.iou)

        result = OrderedDict()
        if detections is not None:
            detections = rescale_boxes(detections[0], self.img_size, shape)
            detections = detections.numpy().tolist()
            out_classes = [x[6] for x in detections]
            out_scores = [x[5] for x in detections]
            out_boxes = [x[:4] for x in detections]

            detection_class_names = []
            for class_id in out_classes:
                class_name = self.classes[int(class_id)]
                class_name = self.label_map[class_name] + '/' + class_name
                detection_class_names.append(class_name)
            out_boxes_list = []
            for box in out_boxes:
                out_boxes_list.append([round(float(v), 1) for v in box])
            result['detection_classes'] = detection_class_names
            result['detection_scores'] = [round(float(v), 4) for v in out_scores]
            result['detection_boxes'] = out_boxes_list
        else:
            result['detection_classes'] = []
            result['detection_scores'] = []
            result['detection_boxes'] = []

        return result

    def _postprocess(self, data):
        return data

    def inference(self, data):
        '''
        Wrapper function to run preprocess, inference and postprocess functions.

        Parameters
        ----------
        data : map of object
            Raw input from request.

        Returns
        -------
        list of outputs to be sent back to client.
            data to be sent back
        '''
        pre_start_time = time.time()
        data = self._preprocess(data)
        infer_start_time = time.time()
        # Update preprocess latency metric
        pre_time_in_ms = (infer_start_time - pre_start_time) * 1000
        logger.info('preprocess time: ' + str(pre_time_in_ms) + 'ms')

        if self.model_name + '_LatencyPreprocess' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyPreprocess'].update(pre_time_in_ms)

        data = self._inference(data)
        infer_end_time = time.time()
        infer_in_ms = (infer_end_time - infer_start_time) * 1000

        logger.info('infer time: ' + str(infer_in_ms) + 'ms')
        data = self._postprocess(data)

        # Update inference latency metric
        post_time_in_ms = (time.time() - infer_end_time) * 1000
        logger.info('postprocess time: ' + str(post_time_in_ms) + 'ms')
        if self.model_name + '_LatencyInference' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyInference'].update(post_time_in_ms)

        # Update overall latency metric
        if self.model_name + '_LatencyOverall' in MetricsManager.metrics:
            MetricsManager.metrics[self.model_name + '_LatencyOverall'].update(pre_time_in_ms + post_time_in_ms)

        logger.info('latency: ' + str(pre_time_in_ms + infer_in_ms + post_time_in_ms) + 'ms')
        data['latency_time'] = str(round(pre_time_in_ms + infer_in_ms + post_time_in_ms, 1)) + ' ms'
        return data


def parse_classify_rule(json_path=''):
    with codecs.open(json_path, 'r', 'utf-8') as f:
        rule = json.load(f)
    label_map = {}
    for super_label, labels in rule.items():
        for label in labels:
            label_map[label] = super_label
    return label_map
