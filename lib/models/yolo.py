import torch
import logging
import os
import sys
from torch import nn

"""NOTES: Please change the pt_path according to your workspace."""

class DET_YOLO(nn.Module):
    '''
    Detector using yolo v5.
    '''
    def __init__(self, model='yolov5s', single_img=True, img_resize=640, 
                 only_people=True, conf=None, iou=None, verbose=False, pt_path = './det_model/pt') -> None:
        '''
        Initialize the yolo v5 detector.

        Args:
            model (str):        Choose the yolo_v5 model. Defaults to 'yolov5s'.
                                For details on all available models, please see
                                https://github.com/ultralytics/yolov5#pretrained-checkpoints.

            single_img (bool):  If True, only reveive a single image as input. In this case, 
                                return the results as a tensor. Otherwise list. Defaults to True.

            img_resize (int):   Resize scale of input. Defaults to 640.

            only_people (bool): If True, only detect people. Defaults to True.

            conf (float):       Conference threshold (0-1). Defaults to None.

            iou (float):        IoU threshold (0-1). Defaults to None.

            verbose (bool):     If True, show the verbose info of model. Defaults to False.

            pt_path(str):       Path which stores the model pt. Change it before use this class.
        '''

        super(DET_YOLO, self).__init__()
        
        # Disabled verbose info
        if not verbose:
            logging.basicConfig(level='ERROR')

        # Load Model
        self.yolo_model = model
        model_path = os.path.join(pt_path, self.yolo_model+'.pt')
        self.model = torch.hub.load(
            'ultralytics/yolov5', model='custom', path=model_path, verbose=verbose)

        # Model Settings
        self.single_img = single_img
        self.img_resize = img_resize
        self.model.classes = [0] if only_people else None

        if conf is not None:
            assert (conf >= 0.0 and conf <= 
                    1.0), 'The conf thresh must be [0,1], given {}.'.format(conf)
            self.model.conf = conf
        if iou is not None:
            assert (iou >= 0.0 and iou <=
                    1.0), 'The conf thresh must be [0,1], given {}.'.format(iou)
            self.model.iou = iou

        
    def forward(self, input_imgs):
        # Format: list - [ [xmin, ymin, xmax, ymax, confidence, class], ... ]
        det_results = self.model(input_imgs, size=self.img_resize).xyxy

        if self.single_img:
            # Format: tensor - [xmin, ymin, xmax, ymax, confidence, class]
            det_results = det_results[0]

        return det_results


def results_numpy(det_results):
    '''
    Convert the tensor results to numpy. Only valid when input is single image.

    Args:
        det_results: detection results as tensor form, NOT list!

    Returns:
        det_results: the numpy array of results (on cpu)
    '''
    if not torch.is_tensor(det_results):
        print('[Warning] Input type is {}, can not convert to numpy.'.format(type(det_results)))
        return det_results
    det_results = det_results.cpu().numpy()
    return det_results
