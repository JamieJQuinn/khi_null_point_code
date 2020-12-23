IMAGE_FOLDER="../../../images/null_point_khi/field_line_plots"
RAW_IMAGES=$(shell ls raw/*.png)
OUTPUT_IMAGES=$(addprefix cropped/,$(notdir $(RAW_IMAGES:%.png=%_cropped.png)))

all: raw

.PHONY: raw
raw: $(IMAGE_FOLDER)
	mkdir -p raw
	cd ./raw;\
	rm -f $(shell find . -size 0);\
	bash ../fill_filelist.sh;\
	visit -nowin -cli -s ../plot.py
	cp -r raw $(IMAGE_FOLDER)

.PHONY: cropped
cropped: $(IMAGE_FOLDER)
	mkdir -p cropped
	$(MAKE) -C . $(OUTPUT_IMAGES)
	cp -r cropped $(IMAGE_FOLDER)

$(IMAGE_FOLDER):
	mkdir -p $(IMAGE_FOLDER)


cropped/%_cropped.png: raw/%.png
	convert $< -trim $@

.PHONY: clean
clean:
	rm -f raw/*.png cropped/*.png raw/visitlog.py raw/filename_list
