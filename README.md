# Cover Letter Agent

A command-line application that turns a job description, your CV, and your writing guidance into an English cover letter. It uses LangGraph to analyse the job, check whether essential facts are missing, prepare an evidence-based outline, draft the letter, and save Markdown and PDF copies. The prompts and interactive questions are in Chinese; the final letter is in English.

This repository contains **blank input templates**. You must supply your own CV, letter guidance, job description, and model credentials before a full run. The model provider must offer an OpenAI-compatible chat API with JSON Schema structured output and support the `enable_thinking` request parameter used by this project.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/) for the commands below
- Access to a compatible LLM endpoint and its API key

## Quick start

```bash
git clone https://github.com/ywan614/cover-letter-agent.git
cd cover-letter-agent
uv sync
cp config.ini.example config.ini
cp src/template_example.py src/template.py
```

Fill the three required model settings in `config.ini`:

| Section | Field | Purpose |
| --- | --- | --- |
| `[openai]` | `OPENAI_API_KEY` | API key for the chosen model endpoint |
| `[model]` | `MODEL_NAME` | Model identifier accepted by that endpoint |
| `[model]` | `MODEL_BASE_URL` | OpenAI-compatible API base URL |

Every field in `config.ini.example` is intentionally blank. The other fields are optional in the current workflow. Blank logging fields use built-in defaults: `log/`, `app.log`, `INFO`, a 10 MiB rotation limit, five backups, and console logging enabled. Blank `TIMEOUT_SECONDS` uses 300 seconds; set a positive finite number if needed. `LANGSMITH_API_KEY` is only loaded when supplied, and tracing is controlled separately by the `LANGSMITH_TRACING` environment variable. `POSTGRES_URI`, `SANDBOX_ID`, and `LOG_VALUE_MAX_LENGTH` are reserved fields that the current workflow does not use.

In your private `src/template.py`, fill these string constants:

- `CV_TEXT`: your actual CV and verified work history; required.
- `SCENES_TEXT`: optional project examples, context, and wording boundaries; it may stay empty.
- `COVER_LETTER_TEMPLATE`: the letter structure and writing instructions; required. Include any contact details only here, in the private file.

Supply a job description in either of two ways:

```bash
# Recommended: read a private UTF-8 text file outside the repository.
uv run main.py --name "ExampleCo__AI-Engineer" --jd-file /path/to/job-description.txt

# Or copy the blank module and fill JD_TEXT in the ignored file.
cp data/jd.example.py data/jd.py
uv run main.py --name "ExampleCo__AI-Engineer"
```

`--name` is optional and defaults to `untitled`. It labels the run directory; it does not affect the letter. Names are cleaned for use as file paths. `--jd-file` takes precedence over `data/jd.py` and can also point to a saved `output/.../jd.txt` for a fresh run.

To check model client and graph setup without calling the model or reading a job description:

```bash
uv run main.py --check
```

The check still requires the three model settings. It does not verify that the remote endpoint accepts a request. A full run calls the model and may incur provider charges.

## What a run does

1. **Analyse the job:** extract its role, company, duties, required skills, and preferred skills.
2. **Evaluate the evidence:** compare the job with your CV, examples, and letter guidance. The agent normally proceeds. It asks for a missing fact only when a truthful, relevant letter cannot otherwise be written.
3. **Prepare an outline:** select relevant evidence and record how it supports each part of the letter.
4. **Generate the English letter:** use the outline and original inputs to check factual claims and polish the wording.
5. **Save files:** write `cover_letter.md` and `cover_letter.pdf` from the same text.

If prompted for more information, type an answer and press Enter. An empty answer continues with the available evidence. There are at most two rounds of questions. Interactive progress is kept in memory for the current process only; restarting begins a new run.

Each attempt gets a separate directory under `output/`. It contains the original `jd.txt`, `run.json`, `run.log`, token usage, and, after success, `final_state.json`, `cover_letter.md`, and `cover_letter.pdf`. `final_state.json` contains the raw job description, CV, examples, answers, and generated text. Treat the entire directory as private. A failure can leave diagnostic and usage files without final outputs. A forced stop can leave `run.json` marked `running`.

## Run individual stages

`test.py` uses the real graph and model, ending after the selected stage. Upstream stages still run, so these commands can incur API charges:

```bash
uv run test.py --stop-after analyze_jd --jd-file /path/to/job-description.txt
uv run test.py --stop-after evaluate --jd-file /path/to/job-description.txt
uv run test.py --stop-after prepare --jd-file /path/to/job-description.txt
uv run test.py --stop-after generate_output --jd-file /path/to/job-description.txt
uv run test.py --stop-after save_output --jd-file /path/to/job-description.txt
```

Without `--stop-after`, `test.py` stops after `analyze_jd`. Add `--name` to any of these commands to label its output directory. The offline unit tests use mocked model responses and need no credentials:

```bash
uv run python -m unittest discover -s tests
```

## Privacy and publishing

`config.ini`, `src/template.py`, `data/jd.py`, logs, generated outputs, virtual environments, and local environment files are ignored by Git. The tracked `.example` files contain blank values. The repository contains no real CV, job posting, API key, or generated letter. Keep job descriptions, personal facts, and run results out of commits. Before publishing your own fork or changes, inspect the staged file list and diff:

```bash
git diff --cached --name-only
git diff --cached
```

Ignoring a path does not remove it from an earlier Git commit. If you have committed a credential elsewhere, remove it from history and rotate it before publishing.

## Project layout

| Path | Role |
| --- | --- |
| `main.py` | Full command-line workflow |
| `test.py` | Run through a selected graph stage |
| `src/graph.py`, `src/nodes/` | LangGraph routes, model calls, and output rendering |
| `src/state.py` | Shared workflow state |
| `src/run_files.py`, `src/usage.py` | Private run archives and token accounting |
| `src/template_example.py`, `data/jd.example.py` | Blank private-input templates |
| `config.ini.example` | Blank configuration template |
| `skills/humanizer/` | Writing guidance read during generation, with its license |
| `tests/` | Offline unit tests |

The PDF renderer uses ReportLab's bundled Bitstream Vera font and A4 paper. Unsupported characters or markup cause an explicit error. The tool does not send applications or contact employers.
