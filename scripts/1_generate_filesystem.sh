
start=0
end=25

model="Qwen/Qwen3-4B-Instruct-2507"
device_map=auto

cache_dir=./data

python scripts/1_generate_filesystem.py \
    --start $start --end $end \
    --model $model --device-map $device_map \
    --cache-fs --cache-dir $cache_dir \
    --log-level INFO
