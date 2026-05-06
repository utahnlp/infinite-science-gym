
start=0
end=25

provider=openai
model=gpt-5.4

paraphrase_key=templated

cache_dir=./data

python scripts/4_run_evaluation.py \
    --provider $provider --model $model \
    --paraphrase-key $paraphrase_key --sleep $sleep \
    --skip-existing --fail-on-error \
    --cache-dir $cache_dir
