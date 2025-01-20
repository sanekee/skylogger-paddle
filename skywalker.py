
import math
from typing import Optional, Tuple
import cv2
from aoi import find_aoi
from context import FrameContext
from debug import _debug, _debug_displays, _debug_projection
from display import Digit, Display
from ocr import OCR, OCRResult
from utils import Rect, calculate_projection, find_central_box_index, find_projection_rect_index

class Section:
    def __init__(self, name: str, angle: float, length: float, skip_detect: bool = False):
        self.name = name
        self.angle = angle
        self.length = length
        self.skip_detect = skip_detect

class Result:
    def __init__(self, name: str):
        self.name = name
        self.temperature = 0
        self.profile = ""
        self.power = 0
        self.fan = 0
        self.time = 0
        self.mode = ""
        
class SkyWalker():
    def __init__(self, ctx: FrameContext):
        self.ctx = ctx

        self.__init_sections()
        self.minAreaSize = 50

    def __init_sections(self):
        self.__sections = {
            "TEMPERATURE": Section("TEMPERATURE", -149.85, 4.91),
            "PROFILE": Section("PROFILE", -51.16, 2.92),
            "POWER": Section("POWER", 0, 0),
            "FAN": Section("FAN", 0.0, 4.67),
            "TIME": Section("TIME", 165.21, 4.48),
            "MODE_PREHEAT": Section("MODE_PREHEAT", 113.12, 4.24, True),
            "MODE_ROAST": Section("MODE_ROAST", 84.91, 3.6, True),
            "MODE_COOL": Section("MODE_COOL", 54.85, 4.77, True),
        }

    def __preprocess_image(self) -> cv2.Mat:
        ctx = self.ctx
        gray_image = cv2.cvtColor(ctx.image, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)) 
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        _, threshold_image = cv2.threshold(dilated_image, 200, 255, cv2.THRESH_BINARY) 

        return threshold_image

    def __detect_displays2(self, results: list[OCRResult]) -> list[Display]:
        if not results or len(results) == 0:
            return None

        max_height = max([res.box[3] for res in results])
        results = [res for res in results if res.box[3] >= 0.4 * max_height]
        displays: dict[str, Display]= {}

        cidx = find_central_box_index([Rect(res.box) for res in results])
        res = results[cidx]

        display = Display(self.ctx, 'POWER', Rect(res.box), None)
        display.value = res.value

        displays['POWER'] = display
        
        rects = [Rect(res.box) for res in results]
        
        _debug(self.ctx, lambda: _debug_projection(self.ctx, rects))

        def __debug_distance():
            img = self.ctx.image.copy()
            
            cv2.circle(img, Rect(res.box).projected().center(), 3, (0, 255, 0), 3)

            pt_checks = []
            for section in self.__sections.values():
                if section.name == 'POWER':
                    continue

                pt_check = calculate_projection(Rect(res.box), section.length, section.angle)
                pt_checks.append(pt_check)
                cv2.circle(img, pt_check, 3, (0, 0, 255), 3)

            for rect in rects:

                if rect == Rect(res.box):
                    continue
                cv2.rectangle(img, rect.to_list(), (0, 0, 255), 1)
                pt2 = rect.projected().center()
                cv2.rectangle(img, rect.projected().to_list(), (0, 0, 255), 1)

                cv2.circle(img, pt2, 3, (0, 255, 255), 2)
                min_distance = 9999999999999
                pt = []
                for pt_check in pt_checks:
                    distance = int(math.sqrt(((pt_check[0] - pt2[0]) ** 2) + ((pt_check[1] - pt2[1]) ** 2)))
                    min_distance = min(min_distance, distance)
                    if distance == min_distance:
                        pt = pt_check

                cv2.line(img, pt, pt2, (0, 255,255), 1)
                cv2.putText(img, f'{min_distance} - {rect.h * 2}', [pt2[0] + 5, pt2[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            self.ctx._write_step(f'distances', img)
                
        _debug(self.ctx, lambda: __debug_distance())


        for section in self.__sections.values():
            if section.name == 'POWER':
                continue

            pt_check = calculate_projection(Rect(res.box), section.length, section.angle)
            idx2 = find_projection_rect_index(pt_check, rects)

            if idx2 is None:
                continue

            res2 = results[idx2]

            display = Display(self.ctx, section.name, Rect(res2.box), None)

            display.skip_detect = section.skip_detect
            display.value = res2.value

            displays[section.name] =  display

        return displays
        
    @staticmethod
    def __parse_time(time_str: str) -> int:
        if time_str == "----":
            return 0

        if len(time_str) == 5:
            time_str = time_str.replace(':', '')

        if len(time_str) != 4 or not time_str.isdigit():
            raise ValueError(f'invalid time {time_str}')
        
        minutes = int(time_str[:2])
        seconds = int(time_str[2:])

        if minutes >= 60:
            raise ValueError(f'invalid minutes {minutes}')
        
        if seconds >= 60:
            raise ValueError(f'invalid seconds {seconds}')
        
        total_seconds = minutes * 60 + seconds
        return total_seconds

    def detect(self) -> Optional[Result]:
        self.ctx._write_step(f'frame', self.ctx.image)

        results = OCR().detect_panel(self.ctx, self.ctx.image)
        
        def __debug_results():
            img = self.ctx.image.copy()
            for res in results:
                box = res.box
                cv2.rectangle(img, box, (0,255,0),1)
                cv2.putText(img, f'{res.value}', [box[0], box[1] - 20], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            self.ctx._write_step(f'aois', img) 

        _debug(self.ctx, lambda: __debug_results())

        displays = self.__detect_displays2(results)

        if not displays:
            print('skywalker display not found')
            return None

        if not 'POWER' in displays:
            print('skywalker power display not found')
            return None
        
        _debug(self.ctx, lambda: _debug_displays(self.ctx, {key: disp.rect for key, disp in displays.items()}))
                
        orig_res : dict[str, str] = {}
        
        res:Result = Result(self.ctx.name)
        for display in displays.values():
            value = display.value
            if not display.skip_detect:
                _debug(self.ctx, lambda: print(f'{self.ctx.name}-{display.name}: {value}'))
            else:
                if display.name in ["MODE_PREHEAT", "MODE_ROAST", "MODE_COOL"]:
                    value = display.name.removeprefix('MODE_')

            orig_res[display.name] = value
            try:
                match display.name:
                    case "TEMPERATURE":
                        res.temperature = int(value)
                    case "POWER":
                        res.power = int(value)
                    case "FAN":
                        res.fan = int(value)
                    case "TIME":
                        res.time = SkyWalker.__parse_time(value)
                    case "PROFILE":
                        res.profile = value
                    case "MODE_PREHEAT" | "MODE_ROAST" | "MODE_COOL":
                        res.mode = value

            except ValueError as e:
                print(f'{self.ctx.name} - {display.name} failed to convert result ({value}): {e}')

        def _write_diag():
            img = self.ctx.image.copy()
            for display in displays.values():
                box = display.rect.to_list()
                cv2.rectangle(img, box, (0, 0, 255), 1)

                cv2.putText(img, f'{display.name}: {orig_res[display.name]}', [box[0], box[1] - 20], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                
            self.ctx._write_step(f'result', img)

        _debug(self.ctx, lambda: _write_diag())


        return res
