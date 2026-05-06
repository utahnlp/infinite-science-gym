"""
Main evaluation runner script.
Orchestrates evaluations across multiple filesystems and models.
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List

from utils import infer_fs_seeds, setup_logging

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulator.qa import QuestionAnswerPair
from eval.harness import EvaluationHarness

def run_evaluation_final(
    keys: List[str],
    qa_dir: str,
    harness: EvaluationHarness,
    model_results_dir: str,
    paraphrase_key: str = 'templated',
    skip_existing: bool = True,
    fail_on_error: bool = False,
    sleep: int = 0,
    logger = None):
    
    if logger is None:
        logger = logging.getLogger(__name__)

    for i, key in enumerate(keys):
        fs_seed, qa_name = key.split('.')
        fs_seed = int(fs_seed)
        qa_file = os.path.join(qa_dir, f'{fs_seed}.json')
        with open(qa_file) as f:
            d = json.load(f)
        qa_pair = QuestionAnswerPair.from_json(d[qa_name])

        # Check if result already exists and skip if requested
        result_exists = check_result_exists(
            model_results_dir=model_results_dir, 
            fs_seed=fs_seed, 
            qa_name=qa_name)
        if skip_existing and result_exists:
            logger.info(f"{i:3d}/{len(keys)}  [{fs_seed}.{qa_name}] Result already exists, skipping")
            continue

        logger.info(f"{i:3d}/{len(keys)}  [{fs_seed}.{qa_name}]")
        logger.debug(f"  Templated Question: {qa_pair.question.question}")
        if paraphrase_key == 'templated':
            question = qa_pair.question.swap_variables(qa_pair.question.question)
        else:
            if paraphrase_key not in qa_pair.paraphrases:
                raise ValueError(f'Paraphrase {paraphrase_key} not available for {fs_seed}.{qa_name}')
            question = qa_pair.question.swap_variables(qa_pair.paraphrases[paraphrase_key]['paraphrase'])

        logger.info(f"  Question: {question}")
        logger.info(f"  Answer: {qa_pair.answer.answer} (has_answer={qa_pair.answer.has_answer})")

        # Run evaluation
        result = harness.evaluate(fs_seed=fs_seed, question=question, fail_on_error=fail_on_error)
        
        # Log result
        if result['status'] == 'success':
            duration = result.get('duration_seconds', 0)
            logger.info(f"  ✓ Success ({duration:.1f}s)")
        else:
            logger.error(f"  ✗ Error: {result.get('error', 'Unknown error')}")
        
        # Save result
        save_result(
            model_results_dir=model_results_dir, 
            fs_seed=fs_seed, 
            qa_name=qa_name, 
            result=result)

        logger.info(f'sleep for {sleep}s')
        time.sleep(sleep)


def run_evaluation_all_seeds(
    fs_seeds: List[int],
    qa_dir: str,
    harness: EvaluationHarness,
    model_results_dir: str,
    paraphrase_key: str = 'templated',
    skip_existing: bool = True,
    fail_on_error: bool = False,
    sleep: int = 0,
    logger = None):
    """
    Run evaluation for a single filesystem seed.
    
    Args:
        fs_seed: Filesystem seed
        qa_dir: Path to QA directory
        harness: EvaluationHarness instance
        results_file: Path to results file
        skip_existing: Skip if result already exists
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    for fs_seed in fs_seeds:
        # Load QA pairs
        qa_file = os.path.join(qa_dir, f'{fs_seed}.json')
        with open(qa_file, 'r') as f:
            qa_data = json.load(f)
        
        qa_pairs = {name: QuestionAnswerPair.from_json(qa) for name, qa in qa_data.items()}
            
        for qa_name, qa_pair in qa_pairs.items():

            # Check if result already exists and skip if requested
            result_exists = check_result_exists(
                model_results_dir=model_results_dir, 
                fs_seed=fs_seed, 
                qa_name=qa_name)
            if skip_existing and result_exists:
                logger.info(f"[{fs_seed}.{qa_name}] Result already exists, skipping")
                continue
            
            logger.info(f"[{fs_seed}.{qa_name}]")
            logger.debug(f"  Templated Question: {qa_pair.question.question}")
            if paraphrase_key == 'templated':
                question = qa_pair.question.swap_variables(qa_pair.question.question)
            else:
                if paraphrase_key not in qa_pair.paraphrases:
                    raise ValueError(f'Paraphrase {paraphrase_key} not available for {fs_seed}.{qa_name}')
                question = qa_pair.question.swap_variables(qa_pair.paraphrases[paraphrase_key]['paraphrase'])

            logger.info(f"  Question: {question}")
            logger.info(f"  Answer: {qa_pair.answer.answer} (has_answer={qa_pair.answer.has_answer})")

            # Run evaluation
            result = harness.evaluate(fs_seed=fs_seed, question=question, fail_on_error=fail_on_error)
            
            # Log result
            if result['status'] == 'success':
                duration = result.get('duration_seconds', 0)
                logger.info(f"  ✓ Success ({duration:.1f}s)")
            else:
                logger.error(f"  ✗ Error: {result.get('error', 'Unknown error')}")
            
            # Save result
            save_result(
                model_results_dir=model_results_dir, 
                fs_seed=fs_seed, 
                qa_name=qa_name, 
                result=result)
        
            logger.info(f'sleep for {sleep}s')
            time.sleep(sleep)
        
        logger.info(f"Completed evaluation for seed {fs_seed}")


def check_result_exists(model_results_dir: str, fs_seed: int, qa_name: str) -> bool:
    results_file = os.path.join(model_results_dir, f'{fs_seed}.{qa_name}.json')
    if not os.path.exists(results_file):
        return False
    return True


def save_result(model_results_dir: str, fs_seed: int, qa_name: str, result: Dict[str, Any]):
    results_file = os.path.join(model_results_dir, f'{fs_seed}.{qa_name}.json')
    # Write back to file
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run LLM evaluations on filesystem QA pairs.")
    # Cache and data paths
    parser.add_argument(
        "--cache-dir", type=str, default="./data", 
        help="Path to cache directory containing fs/ and qa/ subdirectories")
    parser.add_argument(
        "--qa-dir", type=str, default="qa", 
        help="Subdirectory within cache-dir containing QA files")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Subdirectory within cache-dir where results will be saved")
    parser.add_argument(
        "--key-file", type=str, default=None,
        help=""
    )
    # Seed range
    parser.add_argument(
        "--start", type=int, default=None,
        help="First filesystem seed (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, 
        help="Last filesystem seed (inclusive)")
    # Model configuration
    parser.add_argument(
        "--provider", type=str, default="openai", choices=["openai", "anthropic", "google", "local"],
        help="LLM provider (openai, anthropic, google, local)")
    parser.add_argument(
        "--model", type=str, default="gpt-5.2",
        help="Model name (e.g., 'gpt-5.2', 'claude-3-sonnet', etc.)")
    parser.add_argument(
        "--device-map", type=str, default="auto",
        help="device-map for local models (cuda, cpu, mps, etc.)")
    # Evaluation options
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip evaluation if result already exists (default: True)")
    parser.add_argument(
        "--no-skip-existing", dest="skip_existing", action="store_false",
        help="Force re-evaluation even if result exists")
    parser.add_argument(
        "--paraphrase-key", type=str, default='templated',
        help="Key to the paraphrase to be used for evaluation.")
    parser.add_argument(
        "--fail-on-error", action="store_true", default=False,
        help="Quit if an error is raised during API calls or local forward passes")
    # Logging
    parser.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level")
    parser.add_argument('--sleep', type=int, default=0)
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Paraphrase Key: {args.paraphrase_key}")
    
    # Setup paths
    qa_dir = os.path.join(args.cache_dir, args.qa_dir)
    results_dir = os.path.join(args.cache_dir, args.results_dir)
    
    os.makedirs(results_dir, exist_ok=True)
    
    # Create results file (organized by provider_model)
    def get_simple_key(paraphrase_key):
        if paraphrase_key == 'templated':
            return paraphrase_key
        return paraphrase_key.split('/')[1]
    model_str = args.model if '/' not in args.model else args.model.split('/')[-1]
    model_results_dir = os.path.join(results_dir, f"{args.provider}_{model_str}", get_simple_key(args.paraphrase_key))
    os.makedirs(model_results_dir, exist_ok=True)
    
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info(f"QA dir: {qa_dir}")
    logger.info(f"Results dir: {model_results_dir}")
    
    # Infer filesystem seeds
    if args.key_file:
        logger.info(f'loading QA keys from {args.key_file}')
        with open(args.key_file) as f:
            keys = f.readlines()
            keys = [k.strip() for k in keys if k.strip()]
        logger.info(f'  {len(keys)} keys loaded')
    else:
        fs_seeds = infer_fs_seeds(qa_dir, args.start, args.end)
        if not fs_seeds:
            logger.error("No filesystem seeds found")
            exit()
                
        logger.info(f"Found {len(fs_seeds)} filesystem(s)")
        logger.info(f"  Range: {args.start if args.start is not None else 'unbounded'} to {args.end if args.end is not None else 'unbounded'}")

    # Initialize evaluation harness
    try:
        harness = EvaluationHarness(
            provider=args.provider,
            model_name=args.model,
            device_map=args.device_map)
    except Exception as e:
        logger.error(f"Failed to initialize evaluation harness: {e}")
        return 1
    
    # Run evaluations
    try:
        if args.key_file:
            run_evaluation_final(
                keys=keys,
                qa_dir=qa_dir,
                harness=harness,
                model_results_dir=model_results_dir,
                paraphrase_key=args.paraphrase_key,
                skip_existing=args.skip_existing,
                fail_on_error=args.fail_on_error,
                sleep=args.sleep,
                logger=logger)
        else:
            run_evaluation_all_seeds(
                fs_seeds=fs_seeds,
                qa_dir=qa_dir,
                harness=harness,
                model_results_dir=model_results_dir,
                paraphrase_key=args.paraphrase_key,
                skip_existing=args.skip_existing,
                fail_on_error=args.fail_on_error,
                sleep=args.sleep,
                logger=logger)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        harness.provider_obj.shutdown()
        return 130
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        harness.provider_obj.shutdown()
        return 1
    
    logger.info("Evaluation completed successfully")
    harness.provider_obj.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
