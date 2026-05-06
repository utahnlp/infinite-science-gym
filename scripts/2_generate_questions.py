import argparse
import json
import os
import pathlib
import sys
from typing import Dict

from utils import infer_fs_seeds, setup_logging

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulator import CacheOnlySimulator
from simulator.config import FileConfig    
from simulator.qa import QAConfig, QAGenerator


def generate_questions(fs_seed: int, sim: CacheOnlySimulator) -> Dict[str, dict]:
    fs, status = sim.get_filesystem(fs_seed)
    logger.debug(f'Title: {fs.story.title}')
    logger.debug(f'Description: {fs.story.description}')
    logger.debug('-' * 80)

    generator = QAGenerator(qa_config=QAConfig(), sim=sim, fs=fs)

    det_qa_fn_map = {
        'has_readme': generator.has_readme,
        'get_title': generator.get_title,
        'get_abstract': generator.get_abstract,
        'file_extension': generator.file_extension,
    }
    
    seeded_qa_fn_map = {
        'count_files_prefix': generator.count_files_prefix,
        'count_files_conditions': generator.count_files_conditions,
        'count_rows': generator.count_rows,
        'univariate_statistic_single_file': generator.univariate_statistic_single_file,
        'univariate_statistic_conditions': generator.univariate_statistic_conditions,
        'bivariate_statistic': generator.bivariate_statistic,
        'bivariate_hypothesis': generator.bivariate_hypothesis,
    }

    
    qa_pairs = {}
    for name, fn in det_qa_fn_map.items():
        qa_pair = fn()
        qa_pairs[name] = qa_pair.to_json()
        logger.info(f'{name}()')
        logger.debug(f'Question: {qa_pair.question}')
        logger.debug(f'Answer: {qa_pair.answer}')
        logger.debug('-' * 80)

    for name, fn in seeded_qa_fn_map.items():
        for seed in range(args.n_seeded_samples):
            logger.info(f'{name}(seed={seed})')
            qa_pair = fn(seed=seed)
            qa_pairs[f'{name}_{seed}'] = qa_pair.to_json()
            logger.debug(f'Question: {qa_pair.question}')
            logger.debug(f'Answer: {qa_pair.answer}')
            logger.debug('-' * 80)

    return qa_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full file system builder pipeline.")
    parser.add_argument("--start", type=int, default=None, help="First file system seed.")
    parser.add_argument("--end", type=int, default=None, help="First file system seed.")
    parser.add_argument("--cache-dir", type=str, default="./data", help="Path to cache directory.")
    parser.add_argument("--fs-dir", type=str, default="fs", help="Path to fs directory.")
    parser.add_argument("--qa-dir", type=str, default="qa", help="Path to qa directory.")
    parser.add_argument("--n-seeded-samples", type=int, default=5)
    parser.add_argument(
        "--log-level", type=str, default="INFO", 
        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")

    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    fs_dir = os.path.join(args.cache_dir, args.fs_dir)
    qa_dir = os.path.join(args.cache_dir, args.qa_dir)
    os.makedirs(qa_dir, exist_ok=True)

    sim = CacheOnlySimulator(cfg={'source': 'local', 'cache_dir': fs_dir}, file_config=FileConfig())
        
    # Infer filesystem seeds
    fs_seeds = infer_fs_seeds(fs_dir, args.start, args.end)
    if not fs_seeds:
        logger.error("No filesystem seeds found")
        exit()
            
    logger.info(f"Found {len(fs_seeds)} filesystem(s)")
    if args.start is not None or args.end is not None:
        logger.info(f"  Range: {args.start or 'unbounded'} to {args.end or 'unbounded'}")


    for fs_seed in sorted(fs_seeds):
        logger.info(f'{fs_seed=}')

        qa_pairs = generate_questions(fs_seed, sim)
        
        qa_file = os.path.join(qa_dir, f'{fs_seed}.json')
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=4)
        logger.info(f'written to file: {qa_file}')

        
