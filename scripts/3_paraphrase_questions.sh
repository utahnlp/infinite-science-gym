
start=0
end=25

model="openai/gpt-oss-20b"
device_map="auto"

cache_dir=./data

python scripts/3_paraphrase_questions.py \
    --start $start --end $end \
    --model $model --device-map $device_map \
    --cache-dir $cache_dir --fs-dir fs --qa-dir qa \
    --replace-existing \
    --log-level INFO
