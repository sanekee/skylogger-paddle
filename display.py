
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

    def detect(self) -> str:
        self.ctx._write_step(f'{self.name}', self.__image)

        res_str = OCR().detect(self.ctx, self.name, self.__image)

        return res_str

