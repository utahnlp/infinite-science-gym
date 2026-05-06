# InfiniteScienceGym

Official code for [**InfiniteScienceGym: An Unbounded, Procedurally-Generated Benchmark for Scientific Analysis**](https://arxiv.org/abs/2604.13201).

## Repository Structure

- `./scripts` - Numbered evaluation pipeline scripts:
  - `1_generate_filesystem.py` - Create and cache seeded filesystem metadata
  - `2_generate_questions.py` - Generate templated QA pairs from filesystems
  - `3_paraphrase_questions.py` - Generate paraphrases of QA pairs
  - `4_run_evaluation.py` - Evaluate LLMs on QA pairs
 
- `./simulator` - Core simulation engine:
  - `simulator.py` - Main orchestrator for generating filesystems
  - `story.py` - Scientific research narrative generation
  - `file.py`, `filesystem.py` - File and directory structure generation
  - `qa/` - Question-answer pair generation and paraphrasing
  - `types/` - Core data structures (`Story`, `FileSystem`, `FileVariable`, etc.)
  
- `./taxonomy` - JSON files defining scientific domains (computer science, engineering, life sciences, natural sciences, social sciences, statistics)
  
- `./eval` - Evaluation harness and MCP servers:
  - `harness.py` - Main evaluation orchestrator
  - `providers/` - LLM provider implementations (Anthropic, OpenAI, local)
  - `mcp_servers/` - Model Context Protocol servers:
    - `scientific_data_repository/` - File access and repository tools
    - `python_interpreter/` - Code execution sandbox

## Installation

We run our experiments with a virtual environment using Python 3.12, but this repository should be compatible with any Python version 3.8 or higher.

```bash
git clone https://github.com/utahnlp/infinite-science-gym
cd infinite-science-gym
pip install -r requirements.txt
```

In addition to Python, to run the full evaluation, make sure to install Docker and set up API keys for Anthropic with the environment variable `ANTHROPIC_API_KEY`, and OpenAI with `OPENAI_API_KEY`.

## Running the Code

### 1. Generate Simulated Filesystems

```bash
python scripts/1_generate_filesystem.py \
  --start 0 --end 10 \
  --cache-fs --cache-dir ./data/fs \
  --model Qwen/Qwen3-4B-Instruct-2507  # Change to your preferred huggingface LLM
```

Creates 11 procedurally-generated scientific repositories (seeds 0-10) and caches them locally in the `./data/fs` directory.

### 2. Generate Question-Answer Pairs

```bash
python scripts/2_generate_questions.py \
  --start 0 --end 10 \
  --cache-dir ./data \
  --n-seeded-samples 5
```

Generates templated QA pairs from each filesystem.

### 3. Paraphrase Questions (Optional)

```bash
python scripts/3_paraphrase_questions.py \
  --start 0 --end 10 \
  --cache-dir ./data \
  --n-paraphrases 3
```

Creates natural language variants of questions.

### 4. Run Evaluation

First, set up the MCP servers available to the LLM. In separate terminals:

```bash
# Terminal 1: Scientific Data Repository (port 8000)
docker build -t scientific-data-repository-image -f mcp_scientific_data_repository.Dockerfile .
docker run --rm -p 8000:8000 scientific-data-repository-image

# Terminal 2: Python Interpreter (port 8001)
docker build -t python-interpreter-image -f mcp_python_interpreter.Dockerfile .
docker run --rm -p 8001:8000 python-interpreter-image
```

Then run the evaluation:

```bash
python scripts/4_run_evaluation.py \
  --provider anthropic \
  --model claude-3-5-sonnet \
  --start 0 --end 10 \
  --qa-dir ./data/qa
```

Results will be saved with LLM responses and evaluation metrics. The `./eval/providers/` directory supports multiple LLM backends:

```bash
python scripts/4_run_evaluation.py --provider openai --model gpt-4o ...
python scripts/4_run_evaluation.py --provider anthropic --model claude-3-sonnet ...
python scripts/4_run_evaluation.py --provider local --model Qwen/Qwen3-32B-Instruct ...
```

If evaluating OpenAI or Anthropic models, make sure to set up API keys with the environment variable `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`, respectively.

## Programmatic Usage

### Access Data Directly

```python
from simulator import Simulator
from simulator.config import StoryConfig, FileSystemConfig, FileConfig
from simulator.utils import load_taxonomy

# Initialize simulator
sim = Simulator(
    story_config=StoryConfig(taxonomy=load_taxonomy('./taxonomy')),
    fs_config=FileSystemConfig(),
    file_config=FileConfig()
)

# Generate a filesystem for seed 42
# Seed 42 always produces the same filesystem and data
fs = sim.get_filesystem(seed=42)
print(f"Repository: {fs.story.title}")
print(f"# Files: {len(fs.tree.get_paths())}")

# Read a file
first_file_path = fs.tree.get_paths()[0]
print(f"Path: {first_file_path}")
file = sim.read_file(fs=fs, path=first_file_path)
print(file.data.head(10)) # file.data is a pandas DataFrame
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

If you use InfiniteScienceGym in your research, please cite:

```bibtex
@misc{bentham2026infinitesciencegym,
      title={InfiniteScienceGym: An Unbounded, Procedurally-Generated Benchmark for Scientific Analysis}, 
      author={Oliver Bentham and Vivek Srikumar},
      year={2026},
      eprint={2604.13201},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.13201}, 
}
```
