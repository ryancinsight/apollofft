import os
import re

# List of prohibited cosmetic words based on the gap audit
adverbs = [
    "natively", "correctly", "efficiently", "isolating", "cleanly", "dynamically",
    "mathematically", "mapping", "structurally", "rationally", "explicitly",
    "identically", "precisely", "organically", "seamlessly", "flawlessly",
    "properly", "flexibly", "intelligently", "purely", "optimally", "safely",
    "symmetrically", "uniformly", "squarely", "statically", "smoothly",
    "perfectly", "securely", "clearly", "reliably", "nicely", "compactly",
    "fully", "smartly", "homogeneously", "firmly", "accurately", "evenly",
    "neatly", "effectively", "exactly", "strongly", "robustly", "successfully",
    "completely", "softly", "linearly", "cleverly", "strictly", "rapidly",
    "tightly", "carefully", "solidly", "confidently", "snugly", "identical",
    "gracefully", "functionally", "elegantly", "coherently", "expertly",
    "physically", "logically", "strictly", "synchronously"
]

adverb_regex = re.compile(r'\b(?:' + '|'.join(adverbs) + r')\b', flags=re.IGNORECASE)

def clean_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    changed = False
    
    def process_line(line):
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!") or stripped.startswith("//"):
            # Clean adverbs
            cleaned = adverb_regex.sub("", line)
            # Remove repeated spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            # Fix cases where we stripped too much and got "/// "
            if cleaned.strip() == "///" or cleaned.strip() == "//!":
                cleaned = line[:line.find('/')+3]
            
            # Clean up trailing spaces before punctuation
            cleaned = cleaned.replace(" .", ".").replace(" ,", ",")
            return cleaned
        return line

    for i in range(len(lines)):
        original = lines[i]
        lines[i] = process_line(lines[i])
        if lines[i] != original:
            changed = True
            
    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Cleaned {path}")

for root, dirs, files in os.walk(r"d:\apollofft\crates"):
    for file in files:
        if file.endswith(".rs"):
            clean_file(os.path.join(root, file))
