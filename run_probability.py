"""Probability that a uniformly random bit sequence of length N contains
at least K consecutive zeroes.

Exact answer via dynamic programming over the length of the current
trailing run of zeroes (a Markov chain with K+1 states, the last absorbing),
computed with exact rational arithmetic. A Monte Carlo simulation is run
as an independent sanity check.

With --llm, it also asks Claude Opus 5.5 for LLM_SEQUENCES "random"
sequences in a single request and compares how often they contain the run
with the exact probability (requires `pip install anthropic` and API
credentials, e.g. ANTHROPIC_API_KEY).
"""

import argparse
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction

N = 300
K = 8
LLM_SEQUENCES = 100
LLM_MODEL = "claude-opus-5-5"
LLM_RAW_OUTPUT = "llm_sequences.txt"


def exact_probability(n: int, k: int) -> Fraction:
    # state[j] = probability that the sequence so far has NOT yet contained
    # k zeroes in a row and currently ends with exactly j zeroes (0 <= j < k).
    half = Fraction(1, 2)
    state = [Fraction(0)] * k
    state[0] = Fraction(1)
    for _ in range(n):
        new = [Fraction(0)] * k
        new[0] = sum(state) * half          # next bit is 1: run resets
        for j in range(k - 1):
            new[j + 1] = state[j] * half    # next bit is 0: run grows
        # state[k-1] * half flows into the absorbing "found a run" state
        state = new
    return 1 - sum(state)


def count_sequences_without_run(n: int, k: int) -> int:
    # Cross-check: number of length-n bit strings with no k-run of zeroes
    # follows the k-bonacci recurrence a(n) = a(n-1) + ... + a(n-k),
    # with a(m) = 2**m for 0 <= m < k.
    a = [2**m for m in range(k)]
    for m in range(k, n + 1):
        a.append(sum(a[m - k:m]))
    return a[n]


def monte_carlo(n: int, k: int, trials: int, seed: int = 12345) -> float:
    rng = random.Random(seed)
    target = "0" * k
    hits = 0
    for _ in range(trials):
        bits = format(rng.getrandbits(n), f"0{n}b")
        if target in bits:
            hits += 1
    return hits / trials


def generate_llm_sequences(n: int, count: int) -> str:
    # One request for all sequences; returns the raw text of the reply.
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        f"Generate {count} independent, uniformly random bit sequences, "
        f"each exactly {n} characters long, using only the characters 0 and 1. "
        f"Output one sequence per line and nothing else: no numbering, "
        f"no spaces, no commentary."
    )
    # Streaming because the reply is long (~count * n characters).
    with client.messages.stream(
        model=LLM_MODEL,
        max_tokens=64000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError(f"Model refused: {message.stop_details}")
    if message.stop_reason == "max_tokens":
        print("Warning: reply hit max_tokens; the last sequences may be cut off.")
    return "".join(b.text for b in message.content if b.type == "text")


def longest_shared_stretches(seqs: list[str], seed: int = 40) -> dict:
    # Maximal stretches (>= seed bits) shared by two different sequences,
    # keyed by (line_a, start_a, line_b, start_b) -> length. For random bits
    # a 40-bit match between any two lines is essentially impossible.
    index = defaultdict(list)
    for i, x in enumerate(seqs):
        for j in range(len(x) - seed + 1):
            index[x[j:j + seed]].append((i, j))
    matches = {}
    for occ in index.values():
        for a in range(len(occ)):
            for b in range(a + 1, len(occ)):
                (i, p), (k, q) = occ[a], occ[b]
                if i == k:
                    continue
                while p > 0 and q > 0 and seqs[i][p - 1] == seqs[k][q - 1]:
                    p, q = p - 1, q - 1
                e = 0
                while (p + e < len(seqs[i]) and q + e < len(seqs[k])
                       and seqs[i][p + e] == seqs[k][q + e]):
                    e += 1
                matches[(i, p, k, q)] = e
    return matches


def pattern_report(seqs: list[str], near_copy: int = 250) -> None:
    # Two failure modes of "human-like" randomness: switching between 0 and 1
    # too often, and reusing earlier stretches of output.
    m = len(seqs)
    switches = sum(sum(a != b for a, b in zip(s, s[1:])) for s in seqs)
    print(f"  Switch rate between neighbours: "
          f"{switches / sum(len(s) - 1 for s in seqs):.3f} (ideal 0.5)")

    blocks = Counter(s[i:i + 4] for s in seqs for i in range(0, len(s) - 3, 4))
    expected = sum(blocks.values()) / 16
    print(f"  4-bit block counts (ideal ~{expected:.0f} each):")
    for b in sorted((format(v, "04b") for v in range(16)), key=lambda b: -blocks[b]):
        print(f"    {b}: {blocks[b]:5d}  ({blocks[b] / expected:.2f}x)")

    matches = longest_shared_stretches(seqs)
    if not matches:
        print("  No 40-bit stretch is shared between lines.")
        return
    (i, p, k, q), e = max(matches.items(), key=lambda t: t[1])
    lines_with_copy = {x for (a, _, c, _) in matches for x in (a, c)}
    print(f"  Lines sharing a >=40-bit stretch with another line: "
          f"{len(lines_with_copy)}/{m}")
    print(f"  Longest shared stretch: {e} bits "
          f"(line {i + 1} @{p} = line {k + 1} @{q})")

    parent = list(range(m))
    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for (a, _, c, _), length in matches.items():
        if length >= near_copy:
            parent[find(a)] = find(c)
    groups = defaultdict(list)
    for a in range(m):
        groups[find(a)].append(a + 1)
    copies = sorted((g for g in groups.values() if len(g) > 1), key=len, reverse=True)
    print(f"  Distinct sequences (near-copies sharing >={near_copy} bits merged): "
          f"{len(groups)} of {m}")
    for g in copies:
        print(f"    {len(g)} near-copies: lines {g}")


def llm_experiment(n: int, k: int, count: int, p: Fraction,
                   path: str | None = None) -> None:
    if path:
        with open(path) as f:
            text = f.read()
        source = f"sequences read from {path}"
    else:
        text = generate_llm_sequences(n, count)
        with open(LLM_RAW_OUTPUT, "w") as f:
            f.write(text)
        source = (f"{LLM_MODEL}, one request for {count} sequences "
                  f"(raw reply saved to {LLM_RAW_OUTPUT})")

    candidates = [ln.strip() for ln in text.splitlines()
                  if re.fullmatch(r"[01]+", ln.strip())]
    valid = [s for s in candidates if len(s) == n]
    print(f"\nLLM ({source})")
    print(f"  Bit-string lines returned: {len(candidates)}, "
          f"with correct length {n}: {len(valid)}")
    if not valid:
        return

    m = len(valid)
    hits = sum(("0" * k) in s for s in valid)
    zeros = sum(s.count("0") for s in valid) / (m * n)
    longest = [max(map(len, re.findall("0+", s)), default=0) for s in valid]
    print(f"  Fraction of zero bits:        {zeros:.4f} (ideal 0.5)")
    print(f"  Mean longest run of zeroes:   {sum(longest) / m:.2f}")
    print(f"  Sequences with {k} zeroes run: {hits}/{m} = {hits / m:.4f}")

    # Compare to the exact probability: expected count and a two-sided
    # exact binomial p-value (sum of outcomes no more likely than observed).
    pf = float(p)
    pmf = [math.comb(m, i) * pf**i * (1 - pf)**(m - i) for i in range(m + 1)]
    p_value = min(1.0, sum(q for q in pmf if q <= pmf[hits] * (1 + 1e-9)))
    print(f"  Expected for truly random:    {m * pf:.1f} ± {math.sqrt(m * pf * (1 - pf)):.1f}")
    print(f"  Two-sided binomial p-value:   {p_value:.3g}")

    pattern_report(valid)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", action="store_true",
                        help=f"also test {LLM_SEQUENCES} sequences generated by "
                             f"{LLM_MODEL} in a single request")
    parser.add_argument("--llm-file", metavar="PATH",
                        help="analyse LLM-generated sequences saved in PATH "
                             "(one per line) instead of calling the API")
    args = parser.parse_args()

    p = exact_probability(N, K)
    p_check = 1 - Fraction(count_sequences_without_run(N, K), 2**N)
    assert p == p_check, "DP and combinatorial count disagree"

    print(f"N = {N}, K = {K}")
    print(f"Exact probability:  {float(p):.15f}")
    print(f"As a fraction:      {p.numerator} / 2^{N}"
          if p.denominator == 2**N else f"As a fraction: {p}")

    trials = 1_00
    mc = monte_carlo(N, K, trials)
    se = (mc * (1 - mc) / trials) ** 0.5
    print(f"Monte Carlo ({trials:,} trials): {mc:.6f} ± {1.96 * se:.6f} (95% CI)")

    if args.llm_file:
        llm_experiment(N, K, LLM_SEQUENCES, p, path=args.llm_file)
    elif args.llm:
        llm_experiment(N, K, LLM_SEQUENCES, p)


if __name__ == "__main__":
    main()
