from image_segmentation import preprocess_inference_images
roi3_raw = '/home/weismanal/links/1-pre-processing/roi3/1-not_padded/roi3_images_original.npy'
fov_raw = '/home/weismanal/links/1-pre-processing/field_of_view/field_of_view.npy'
preprocess_inference_images(roi3_raw, 2, 6, 'inference_images/roi3_prepared')
preprocess_inference_images(fov_raw, 2, 6, 'inference_images/fov_prepared')