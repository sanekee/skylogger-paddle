
from typing import Tuple

import cv2
from context import FrameContext
from debug import _debug
from ocr import OCR
from utils import Rect

class Digit:
    def __init__(self, ctx: FrameContext, name: str, index: int, rect: Rect):
        self.ctx = ctx
        self.name = name
        self.index = index
        self.rect = rect

        self.sliding = False
        self.max_width = -1
        
class Display:
    def __init__(self, ctx: FrameContext, name: str, rect: Rect, digits: list[Digit]):
        self.ctx = ctx
        self.name = name
        self.rect = rect
        self.digits  = digits
        self.skip_detect = False
        
        self.__extract_image()


    def __extract_image(self):
        self.__image = self.rect.extract_image(self.ctx.image)

    def get_max_digit_size(self) -> Tuple[int, int]:
        max_w = 0
        max_h = 0
        for digit in self.digits:
            max_w = max(digit.rect.w, max_w)
            max_h = max(digit.rect.h, max_h)
        return max_w, max_h
    
    def fix_size(self, width: int, height: int):
        new_x, new_y, new_w, new_h = self.rect.to_list()
        new_x2 = new_x + new_w
        new_y2 = new_x + new_w
        if self.rect.w < width:
            new_x = int(max(0, new_x2 - width))
            new_w = width
        
        new_y = int(max(0, new_y - (height - new_h) / 2))
        new_h = height
        self.rect = Rect([new_x, new_y, new_w, new_h])
        self.__image = self.__extract_image()

    def detect(self) -> str:
        self.ctx._write_step(f'{self.name}', self.__image)

        res_str = OCR().recognize(self.ctx, self.name, self.__image)

        return res_str

