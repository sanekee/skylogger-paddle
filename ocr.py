import cv2
from paddleocr import PaddleOCR
from context import FrameContext
from debug import _debug

class OCR:
    __instance = None 
    __ocr = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__ocr = PaddleOCR(lang='en') 

        return cls.__instance
    
    @classmethod
    def detect_row(cls, ctx: FrameContext, name: str, img: cv2.Mat) -> str:
        result = cls.__ocr.ocr(img, det=False, cls=False)

        def __print_res():
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    print(f'{ctx.name}-{name}: {line[0]}, {line[1]}')
    
        _debug(ctx, lambda: __print_res())
        
        if len(result) == 0 or \
            len(result[0]) == 0 or \
            len(result[0][0]) == 0:
            return ''
        
        res0 = result[0]
        res00 = res0[0]
        return res00[0]
        
    @classmethod
    def detect_panel(cls, ctx: FrameContext, img: cv2.Mat) -> str:
        result = cls.__ocr.ocr(img, det=True, cls=False)

        def __print_res():
            for idx in range(len(result)):
                res = result[idx]
                for lineIdx, line in enumerate(res):
                    print(f'{ctx.name}-{idx}-{lineIdx}: {line[0]}, {line[1]}')
    
        _debug(ctx, lambda: __print_res())
        
        return ''
