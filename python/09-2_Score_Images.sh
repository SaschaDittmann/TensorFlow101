#!/bin/bash
find ./resources/test-images -type f \
	-exec python ./scripts/label_image.py \
	--graph=./outputs/image_retraining/output_graph.pb \
    --labels=./outputs/image_retraining/output_labels.txt \
    --input_layer=Placeholder \
    --output_layer=final_result \
    --input_height=299 \
    --input_width=299 \
    --image={} \;
