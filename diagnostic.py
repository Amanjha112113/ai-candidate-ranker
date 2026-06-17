import json
from collections import Counter
from honeypot_checks import run_honeypot_checks
import config

def main():
    flag_counter = Counter()
    hard_count = 0
    total_candidates = 0

    print("Loading global skill frequencies...")
    # First pass: build global skill freq
    global_skill_freq = {}
    with open(config.CANDIDATES_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            for sk in c.get('skills', []):
                name = sk.get('name', '').lower()
                if name:
                    global_skill_freq[name] = global_skill_freq.get(name, 0) + 1

    print("Running diagnostic on all 100k candidates...")
    # Second pass: run honeypot checks
    with open(config.CANDIDATES_PATH, 'r') as f:
        for line in f:
            if not line.strip(): continue
            c = json.loads(line)
            total_candidates += 1
            
            result = run_honeypot_checks(c, global_skill_freq)
            for flag in result['flags']:
                flag_counter[flag] += 1
            if result['hard']:
                hard_count += 1

            if total_candidates % 10000 == 0:
                print(f"Processed {total_candidates} candidates...")

    print(f"\n--- DIAGNOSTIC RESULTS ({total_candidates} candidates) ---")
    print("Flag counts and trigger rates:")
    for flag, count in flag_counter.most_common():
        print(f"  {flag}: {count} ({count/total_candidates*100:.2f}%)")
    print(f"\nTotal Hard honeypots triggered: {hard_count} ({hard_count/total_candidates*100:.2f}%)")

if __name__ == "__main__":
    main()
