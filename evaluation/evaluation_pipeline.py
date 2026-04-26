"""
Batch evaluation runner that sends every question from 100Questions.txt through the
retrieval pipeline and saves the answers to answers.txt.

The script is resumable: it tracks the last question number written to answers.txt
and skips everything up to that point, so you can safely interrupt and restart it
without losing work or duplicating answers.
"""

import re
import subprocess
from pathlib import Path

# ====== CONFIG ======
QUESTIONS_FILE = "../../evaluation/100Questions.txt"
TARGET_SCRIPT = "retrieval_pipeline.py"
OUTPUT_FILE = "../../evaluation/answers.txt"
PYTHON_EXE = "python"
# ====================


def parse_questions(filepath: str):
    text = Path(filepath).read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*(\d+)\.\s+(.*)$", re.MULTILINE)
    return [(int(n), q.strip()) for n, q in pattern.findall(text)]


def get_last_completed(output_file: str) -> int:
    """Return the highest question number already written to the output file.

    Returns 0 if the file doesn't exist or has no numbered entries yet.
    The main loop uses this to skip already-answered questions on resume.
    """
    path = Path(output_file)
    if not path.exists():
        return 0

    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
    nums = [int(n) for n in pattern.findall(text)]

    return max(nums) if nums else 0


def run_target_script(question: str) -> str:
    """
    Drive retrieval_pipeline.py as a subprocess, feeding it one question via stdin.

    We append "quit\n" after the question so the script's input() loop exits cleanly
    instead of blocking forever waiting for more input. The full stdout is returned
    as the answer string.
    """
    result = subprocess.run(
        [PYTHON_EXE, TARGET_SCRIPT],
        input=question + "\nquit\n",
        text=True,
        capture_output=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return result.stdout.strip()


def main():
    questions = parse_questions(QUESTIONS_FILE)
    last_done = get_last_completed(OUTPUT_FILE)

    output_path = Path(OUTPUT_FILE)

    for number, question in questions:
        if number <= last_done:
            continue

        print(f"Processing {number}")

        try:
            answer = run_target_script(question)
        except Exception as e:
            print(f"Stopped at {number}: {e}")
            break

        with output_path.open("a", encoding="utf-8") as f:
            f.write(f"{number}. {answer}\n")

    print("Done.")


if __name__ == "__main__":
    main()