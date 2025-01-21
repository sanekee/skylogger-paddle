
import csv
import os

import cv2

from ocr import OCRResult

class RecognitionResult:
    def __init__(self, name: str, value: str, box: list[int, int, int, int]):
        self.name = name
        self.value = value
        self.box = box

class RecognitionTraining:
    __instance = None 
    __output_path = ''
    __rec_path = ''
    __img_path = ''
    __rec_file = None

    def __new__(cls, output_path: str = ''):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__output_path = os.path.join(output_path, 'train_data')

            cls.__rec_path = os.path.join(cls.__output_path, 'rec')
            cls.__img_path = os.path.join(cls.__rec_path, 'train')
            os.makedirs(cls.__img_path, exist_ok=True)

            out_file = os.path.join(cls.__rec_path, 'rc_gt_train.txt')
            cls.__rec_file = open(out_file, 'w')

        return cls.__instance

    def close(cls):
        cls.__rec_file.close()

    def write_result(cls, image: cv2.Mat, result: RecognitionResult):
        wrt = csv.writer(cls.__rec_file, delimiter='\t')

        img_path = os.path.join(cls.__img_path, f'{result.name}.png')
        img = image[result.box[1]:result.box[1]+result.box[3], result.box[0]:result.box[0]+result.box[2]].copy()
        cv2.imwrite(img_path, img)

        idx = img_path.index('train_data')
        csv_img_path = img_path[idx:]
        wrt.writerow([csv_img_path, result.value])


