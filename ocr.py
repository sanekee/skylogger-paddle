import cv2
from paddleocr import PaddleOCR
from context import FrameContext
from debug import _debug

class OCRResult:
    def __init__(self, paddle_res):
        if len(paddle_res) != 2 or \
            len(paddle_res[0]) != 4 or \
            len(paddle_res[1]) != 2:
            raise ValueError(f'invalid paddle result {paddle_res}')

        coords = paddle_res[0]
        x, y = coords[0]
        x2, y2 = coords[2]

        self.box = [int(x), int(y), int(x2 - x), int(y2 - y)]
        self.value = paddle_res[1][0]
                
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
    def detect_panel(cls, ctx: FrameContext, img: cv2.Mat) -> list[OCRResult]:
        rec_result = cls.__ocr.ocr(img, det=True, cls=False)

        def __print_res():
            for idx in range(len(rec_result)):
                res = rec_result[idx]
                for lineIdx, line in enumerate(res):
                    print(f'{ctx.name}-{idx}-{lineIdx}: {line[0]}, {line[1]}')
    
        _debug(ctx, lambda: __print_res())

        results: list[OCRResult] = []
        for r in rec_result[0]:
            results.append(OCRResult(r))
        
        return results
