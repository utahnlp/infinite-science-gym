
start=0
end=25

cache_dir=./data

python scripts/2_generate_questions.py \
    --start $start --end $end \
    --cache-dir $cache_dir --fs-dir fs --qa-dir qa \
    --log-level INFO
