"""Build combined corpus from Project Gutenberg books.
Normalizes chapter markers to 'CHAPTER N' format, strips headers/footers.
Filters out tiny chapters (< 100 words) from TOC artifacts."""

import re, os

def extract_body(text):
    """Strip Gutenberg header and footer."""
    start = re.search(r'\*\*\* START OF.*?\*\*\*', text)
    if start:
        text = text[start.end():]
    end = re.search(r'\*\*\* END OF', text)
    if end:
        text = text[:end.start()]
    return text.strip()

def process_book(text, chapter_offset):
    """Split text into chapters, normalize markers, filter tiny chapters."""
    lines = text.split('\n')
    chapters = []
    current_lines = []

    for line in lines:
        stripped = line.strip()
        is_chapter = False

        # Match: CHAPTER I., Chapter I, Chapter 1, CHAPTER I, etc.
        if re.match(r'^(CHAPTER|Chapter)\s+[IVXLC0-9]', stripped):
            is_chapter = True
        # Match: Stave/STAVE (Christmas Carol)
        elif re.match(r'^(STAVE|Stave)\s+[IVXLC]', stripped):
            is_chapter = True
        # Match: PART ONE etc. (Treasure Island)
        elif re.match(r'^PART\s+(ONE|TWO|THREE|FOUR|FIVE|SIX)', stripped):
            is_chapter = True

        if is_chapter:
            if current_lines:
                chapters.append('\n'.join(current_lines))
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        chapters.append('\n'.join(current_lines))

    # Filter: only keep chapters with >= 200 words (removes TOC artifacts)
    result_lines = []
    ch_num = chapter_offset
    for ch_text in chapters:
        word_count = len(re.findall(r'\b[a-zA-Z]+\b', ch_text))
        if word_count >= 200:
            ch_num += 1
            result_lines.append(f'CHAPTER {ch_num}')
            result_lines.append(ch_text)

    return '\n'.join(result_lines), ch_num

os.chdir(os.path.dirname(os.path.abspath(__file__)))

books_all = [
    ('alice.txt', 'Alice in Wonderland'),
    ('looking_glass_raw.txt', 'Through the Looking Glass'),
    ('oz_raw.txt', 'The Wonderful Wizard of Oz'),
    ('peter_pan_raw.txt', 'Peter Pan'),
    ('christmas_carol_raw.txt', 'A Christmas Carol'),
    ('treasure_island_raw.txt', 'Treasure Island'),
]

# Medium corpus: Alice + Looking Glass + Oz (~97K words)
books_medium = [
    ('alice.txt', 'Alice in Wonderland'),
    ('looking_glass_raw.txt', 'Through the Looking Glass'),
    ('oz_raw.txt', 'The Wonderful Wizard of Oz'),
]

import sys
if '--all' in sys.argv:
    books = books_all
    outfile = 'corpus_large.txt'
else:
    books = books_medium
    outfile = 'corpus_medium.txt'

all_text = []
chapter_offset = 0
book_stats = []

for filename, title in books:
    if not os.path.exists(filename):
        print(f"SKIP: {filename} not found")
        continue

    with open(filename, 'r', encoding='utf-8-sig') as f:
        raw = f.read()

    if filename == 'alice.txt':
        body = raw  # Alice already has good formatting
    else:
        body = extract_body(raw)

    ch_before = chapter_offset
    text, chapter_offset = process_book(body, chapter_offset)
    ch_count = chapter_offset - ch_before
    word_count = len(re.findall(r'\b[a-zA-Z]+\b', text))
    book_stats.append((title, ch_count, word_count))

    all_text.append(text)
    all_text.append('')

combined = '\n'.join(all_text)

# Final stats
total_words = len(re.findall(r'\b[a-zA-Z]+\b', combined))
total_chapters = combined.count('\nCHAPTER ') + (1 if combined.startswith('CHAPTER ') else 0)

with open(outfile, 'w', encoding='utf-8') as f:
    f.write(combined)

print(f"Combined corpus: {len(combined):,} chars, {total_words:,} words, {total_chapters} chapters")
print(f"\nPer-book breakdown:")
for title, chs, words in book_stats:
    print(f"  {title}: {chs} chapters, {words:,} words")
print(f"\nOutput: {outfile}")
