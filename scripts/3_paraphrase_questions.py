import argparse
import json
import os
import pathlib
import sys

from utils import infer_fs_seeds, setup_logging

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulator import CacheOnlySimulator
from simulator.config import FileConfig
from simulator.qa import QAConfig, ParaphraseGenerator, QuestionAnswerPair

cache = {}

def paraphrase_questions(
        fs_seed: int, 
        sim: CacheOnlySimulator, 
        paraphrase_generator: ParaphraseGenerator, 
        qa_dir: str,
        n_paraphrases: int,
        replace_existing: bool = True):
    
    fs, status = sim.get_filesystem(fs_seed)

    logger.debug(f'Title: {fs.story.title}')
    logger.debug(f'Description: {fs.story.description}')
    logger.debug('-' * 80)

    qa_file = os.path.join(qa_dir, f'{fs_seed}.json')
    with open(qa_file) as f:
        qa_pairs = {n: QuestionAnswerPair.from_json(qa) for n, qa in json.load(f).items()}       

    no_context_qa_pairs = [
        'has_readme', 'get_title', 'get_abstract', 
        'file_extension', 'count_files_prefix', 'count_rows']
    context_qa_pairs = [
        'count_files_conditions', 'univariate_statistic_single_file', 
        'univariate_statistic_conditions', 'bivariate_statistic', 'bivariate_hypothesis']
    
    all_qa_pairs = set(no_context_qa_pairs) | set(context_qa_pairs)
    
    for name, qa_pair in qa_pairs.items():
        if all(x not in name for x in all_qa_pairs):
            raise Exception(f'{name=} not a known qa template')
        requires_context = any(x in name for x in context_qa_pairs)
        
        for seed in range(n_paraphrases):
            paraphrase_key = f'{paraphrase_generator.llm.model_name}/seed={seed}'
            if not replace_existing and (paraphrase_key in qa_pair.paraphrases):
                logger.info(f"{paraphrase_key} already exists for {name}, won't replace")
            else:
                logger.info(f'{name}() | {seed=} | {qa_pair.question.question}')
                use_cache = False
                if qa_pair.question.question in cache:
                    if paraphrase_key in cache[qa_pair.question.question]:
                        p = cache[qa_pair.question.question][paraphrase_key]['p']
                        e = cache[qa_pair.question.question][paraphrase_key]['e']
                        use_cache = True
                        logger.info('  use cache')
                    else:
                        cache[qa_pair.question.question][paraphrase_key] = {}
                else:
                    cache[qa_pair.question.question] = {}
                
                if not use_cache:
                    p, e = paraphrase_generator.paraphrase(
                        qa_pair=qa_pair, 
                        fs=fs, 
                        seed=seed,
                        requires_context=requires_context)
                    cache[qa_pair.question.question][paraphrase_key] = {'p': p, 'e': e}
                    
                qa_pair.paraphrases[paraphrase_key] = {'paraphrase': p, 'explanation': e}
                logger.info(f'paraphrase({seed=}) | {p}')

        with open(qa_file, 'w') as f:
            json.dump({n: qa.to_json() for n, qa in qa_pairs.items()}, f, indent=4)
        logger.info(f'written to file: {qa_file}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full file system builder pipeline.")
    parser.add_argument("--start", type=int, default=None, help="First file system seed.")
    parser.add_argument("--end", type=int, default=None, help="First file system seed.")
    parser.add_argument("--cache-dir", type=str, default="./data", help="Path to cache directory.")
    parser.add_argument("--fs-dir", type=str, default="fs", help="Path to fs directory.")
    parser.add_argument("--qa-dir", type=str, default="qa", help="Path to qa directory.")
    parser.add_argument("--n-paraphrases", type=int, default=1)
    parser.add_argument("--replace-existing", action='store_true')
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model name.")
    parser.add_argument("--device-map", type=str, default="auto", help="device map for LLM (cpu, cuda, mps, etc.)")
    parser.add_argument(
        "--log-level", type=str, default="INFO", 
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    args = parser.parse_args()
    logger = setup_logging(args.log_level)

    fs_dir = os.path.join(args.cache_dir, args.fs_dir)
    qa_dir = os.path.join(args.cache_dir, args.qa_dir)

    # Infer filesystem seeds
    fs_seeds = infer_fs_seeds(fs_dir, args.start, args.end)
    if not fs_seeds:
        logger.error("No filesystem seeds found")
        exit()
            
    logger.info(f"Found {len(fs_seeds)} filesystem(s)")
    logger.info(f"  Range: {args.start if args.start is not None else 'unbounded'} to {args.end if args.end is not None else 'unbounded'}")

    sim = CacheOnlySimulator(cfg={'source': 'local', 'cache_dir': fs_dir}, file_config=FileConfig())
    
    logger.info(f'Initialize paraphrase generator using model: {args.model}')
    paraphrase_generator = ParaphraseGenerator(
        qa_config=QAConfig(
            paraphrase_model_name=args.model, 
            paraphrase_device_map=args.device_map))

    for fs_seed in sorted(fs_seeds):
        logger.info(f'{fs_seed=}')
        paraphrase_questions(
            fs_seed=fs_seed, 
            sim=sim, 
            paraphrase_generator=paraphrase_generator, 
            qa_dir=qa_dir, 
            n_paraphrases=args.n_paraphrases, 
            replace_existing=args.replace_existing)
    
