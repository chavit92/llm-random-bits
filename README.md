# Can an LLM write random bits?

A random 300-bit sequence contains 8 zeroes in a row with probability **44.2%** (computed exactly). Claude Opus 5.5 was asked, in a single prompt, for 100 random 300-bit sequences. In the 87 usable lines it returned, a run of 8 zeroes appeared **0 times** (expected 38.5 ± 4.6, p ≈ 10⁻²²).

Two problems explain it:

1. **Too much switching.** The model flips to the other bit on 69% of steps instead of 50%, so long runs never form.
2. **Repetition.** After the first few lines, the model mostly rewrote earlier lines shifted by a bit or two; the 87 lines hold only about 38 distinct sequences.

**Live page with charts:** https://chavit92.github.io/llm-random-bits/

## Files

- `index.html`: the write-up with charts.
- `run_probability.py`: the exact probability (dynamic programming with exact fractions, cross-checked by counting), a Monte Carlo check, and the analysis of LLM output.
- `llm_sequences.txt`: the model's raw output (89 lines; 87 have exactly 300 bits).

## Reproduce

```
python3 run_probability.py --llm-file llm_sequences.txt   # analyse the saved output
python3 run_probability.py --llm                          # new run via the Anthropic API (needs `pip install anthropic` and ANTHROPIC_API_KEY)
```

## How the sequences were made

One prompt to a fresh Claude Opus 5.5 agent inside Claude Code, which wrote all lines in one response without using code or a random number generator. The prompt is quoted in full on the page.
