export DEVICE_ID=3
python opensora/sample/sample_v1_3.py \
    --model_path LanguageBind/Open-Sora-Plan-v1.3.0/any93x640x640 \
    --version v1_3 \
    --num_frames 93 \
    --height 352 \
    --width 640 \
    --cache_dir "./" \
    --text_encoder_name_1 google/mt5-xxl \
    --text_prompt examples/sora_refine.txt \
    --ae CausalVAEModel_D4_4x8x8  \
    --ae_path LanguageBind/Open-Sora-Plan-v1.2.0/vae \
    --save_img_path "./sample_videos/prompt_list_0_93x640" \
    --fps 18 \
    --guidance_scale 7.5 \
    --num_sampling_steps 100 \
    --max_sequence_length 512 \
    --sample_method EulerAncestralDiscrete \
    --num_samples_per_prompt 1 \
    --rescale_betas_zero_snr \
    --prediction_type "v_prediction"


    # --ae  \
    # --ae_path /home_host/ddd/workspace/checkpoints/LanguageBind/Open-Sora-Plan-v1.3.0/vae \
    # The model is trained arbitrarily on stride=32. So keep the resolution of the inference a multiple of 32. Frames needs to be 4n+1, e.g. 93, 77, 61, 45, 29, 1 (image).