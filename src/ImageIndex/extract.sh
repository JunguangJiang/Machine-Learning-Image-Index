prefix="../../data/"
python extract_features.py \
  --config_path delf_config_detailed.pbtxt \
  --list_images_path ${prefix}image/imagelist_new.txt\
  --output_dir ${prefix}detailed_features