RUN_DIR=experiments/thesis_experiments/runs/20260508_013500_cub100
MAX_SAMPLES=100
STAGE=cub100
run_detection () 
{ 
    python -m llava.eval.model_cot_det_loader --model-path "$MODEL_PATH" --question-file "$DET_QUESTION_FILE" --image-folder "$IMAGE_FOLDER" --answers-file "$RUN_DIR/raw/detection.jsonl" --temperature 0 --conv-mode "$CONV_MODE" --max-samples "$MAX_SAMPLES" --save-jsonl $([[ "$LOAD_4BIT" == "1" ]] && echo "--load-4bit") > "$RUN_DIR/logs/detection.log" 2>&1
}
run_answer () 
{ 
    local mode="$1";
    shift;
    python -m llava.eval.model_cot_loader "${COMMON_ARGS[@]}" --detection-file "$RUN_DIR/raw/detection.jsonl" --mode "$mode" --answers-file "$RUN_DIR/raw/answer_${mode}.jsonl" "$@" > "$RUN_DIR/logs/answer_${mode}.log" 2>&1
}
