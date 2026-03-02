# regex (L2)
string/regex — Pattern matching (glob match, string escape, word count, regex match/find/replace/test/split)

## Functions
glob_match(s: string, s: string) → float
  Test if string matches glob pattern (1.0/0.0)
str_escape(s: string) → string
  Escape special regex characters in string
word_count(s: string) → int
  Count words in string
re_match(s: string, s: string) → array
  Return all capture groups for first match
re_find(s: string, s: string) → array
  Find all matches of pattern in string
re_replace(s: string, s: string, s: string) → string
  Replace pattern matches with replacement
re_test(s: string, s: string) → float
  Test if pattern matches string (1.0/0.0)
re_split(s: string, s: string) → array
  Split string by regex pattern
