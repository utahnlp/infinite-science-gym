import argparse
import datetime
import json
import os
import pathlib
import sys
import traceback
from datetime import datetime

from utils import setup_logging

project_root = pathlib.Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from simulator import Simulator
from simulator.config import FileConfig, FileSystemConfig, StoryConfig
from simulator.utils import load_taxonomy


def main(args):
    # Create timestamped subdirectory for this run
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_cache_dir = os.path.join(args.cache_dir, timestamp)
    os.makedirs(run_cache_dir, exist_ok=True)
    
    sim = Simulator(
        story_config=StoryConfig(taxonomy=load_taxonomy(args.taxonomy_dir)),
        fs_config=FileSystemConfig(),
        file_config=FileConfig(),
        device_map=args.device_map,
        model_name=args.model)

    seeds = list(range(args.start, args.end+1))
    logger.info(f'seeds set using start={args.start} and end={args.end}')

    for seed in seeds:
        logger.info('\n\n' + '-' * 30 + f' SEED {seed} ' + '-' * 30 + '\n')
        start = datetime.now()
        try:
            if args.field:
                logger.info("Generating story...")
                story = sim.story_generator.generate(
                    seed=seed, 
                    field=args.field, 
                    domain=args.domain, 
                    subdomain=args.subdomain)
                logger.info(f"Generated project title: {story.title}")
                fs = sim.get_filesystem(seed=seed, story=story)
            else:
                fs = sim.get_filesystem(seed)
            
            logger.debug(fs.to_str(exclude_tree=True, exclude_code=True))

            if args.cache_fs:
                path = os.path.join(run_cache_dir, f'{seed}.json')
                with open(path, 'w') as f:
                    json.dump(fs.to_json(), f, indent=2)
                logger.info(f'filesystem cached at: {path}')
        except Exception as e:
            if args.quit_on_error:
                raise e
            traceback.print_exc()
            logger.warning('\n\n' + '-' * 30 + f' FILE SYSTEM w/ SEED {seed} IS CORRUPTED ' + '-' * 30 + '\n')
            if args.cache_fs:
                corrupted_path = os.path.join(run_cache_dir, f'corrupted.txt')
                with open(corrupted_path, 'a') as f:
                    f.write(f'{seed}\n')
        
        dur_min = (datetime.now() - start).total_seconds() / 60
        logger.info(f'Took {dur_min:.1f} minutes')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full file system builder pipeline.")
    parser.add_argument("--start", type=int, default=0, help="First seed.")
    parser.add_argument("--end", type=int, help="Final seed.")
    parser.add_argument("--taxonomy-dir", type=str, default="./taxonomy", help="Path to directory with taxonomies.")
    parser.add_argument("--device-map", type=str, default="auto", help="device map for LLM (cpu, cuda, mps, etc.)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="HuggingFace model name.")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--cache-fs", action="store_true", default=False, help="Will cache generated filesystem if set")
    parser.add_argument("--cache-dir", type=str, default="./data/fs", help="Path to cache directory.")
    parser.add_argument("--field", type=str, default=None, help="Override the randomly chosen field")
    parser.add_argument("--domain", type=str, default=None, help="Override the randomly chosen domain")
    parser.add_argument("--subdomain", type=str, default=None, help="Override the randomly chosen subdomain")
    parser.add_argument("--quit-on-error", action="store_true", default=False, help="")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    main(args)
