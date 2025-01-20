
.PHONY: paddleocr
paddleocr:
	@[ -d "PaddleOCR" ] || \
		git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
