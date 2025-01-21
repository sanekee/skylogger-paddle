CONFIG := data/skywalker_rec.yml
MODEL := PaddleOCR/pretrain_models/en_PP-OCRv4_rec_train/best_accuracy
DICT := data/skywalker_dict.txt

.PHONY: paddleocr
paddleocr:
	@[ -d "PaddleOCR" ] || \
		git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR

train: paddleocr
	FLAGS_allocator_strategy='naive_best_fit' && \
	FLAGS_fraction_of_cpu_memory_to_use='0.4' && \
	python3 PaddleOCR/tools/train.py \
		-c $(CONFIG) \
		-o Global.pretrained_model=$(MODEL) \
		-o Global.character_dict_path=$(DICT)
