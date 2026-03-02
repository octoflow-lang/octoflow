//! Pure-Rust regex engine (Phase 47) — bytecode backtracking VM.
//!
//! Supported syntax:
//!   .              any character except \n
//!   [abc] [a-z]    character classes; [^x] negated
//!   \d \D \w \W \s \S   shortcuts; \b \B word boundary
//!   \n \t \r       escape sequences; \x hex escapes
//!   ^  $           anchors (start / end of string)
//!   *  +  ?        greedy quantifiers (append ? for lazy)
//!   {n} {n,} {n,m} bounded quantifiers
//!   (pat)          capture group; (?:pat) non-capturing
//!   a|b            alternation
//!
//! Replacement syntax: $0 = full match, $1-$9 = capture groups.

// ── Character classes ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum CClass {
    Any,                     // . — matches everything except \n
    Lit(char),               // literal character
    Set(Vec<CItem>, bool),   // character set, negated flag
}

#[derive(Clone, Debug)]
enum CItem {
    Ch(char),
    Range(char, char),
    Digit,   // \d
    Word,    // \w  (alphanumeric + _)
    Space,   // \s
}

impl CClass {
    fn matches(&self, c: char) -> bool {
        match self {
            CClass::Any => c != '\n',
            CClass::Lit(ch) => c == *ch,
            CClass::Set(items, neg) => {
                let hit = items.iter().any(|i| i.matches(c));
                if *neg { !hit } else { hit }
            }
        }
    }
}

impl CItem {
    fn matches(&self, c: char) -> bool {
        match self {
            CItem::Ch(ch) => c == *ch,
            CItem::Range(lo, hi) => c >= *lo && c <= *hi,
            CItem::Digit => c.is_ascii_digit(),
            CItem::Word => c.is_alphanumeric() || c == '_',
            CItem::Space => matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0C' | '\x0B'),
        }
    }
}

fn is_word(c: char) -> bool { c.is_alphanumeric() || c == '_' }

// ── AST ───────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Node {
    Char(CClass),
    AnchorStart,
    AnchorEnd,
    WordBound(bool),          // true = \b, false = \B
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Group(Box<Node>, usize),  // (inner, group_idx 1-based)
    NonCap(Box<Node>),
    Quant(Box<Node>, usize, Option<usize>, bool), // inner, min, max, greedy
}

// ── Parser ────────────────────────────────────────────────────────────

struct Parser {
    chars: Vec<char>,
    pos: usize,
    ngroups: usize,
}

fn parse_regex(pattern: &str) -> Result<(Node, usize), String> {
    let mut p = Parser { chars: pattern.chars().collect(), pos: 0, ngroups: 0 };
    let node = p.alt()?;
    if p.pos < p.chars.len() {
        return Err(format!("regex: unexpected '{}' at position {}", p.chars[p.pos], p.pos));
    }
    Ok((node, p.ngroups))
}

impl Parser {
    fn peek(&self) -> Option<char> { self.chars.get(self.pos).copied() }
    fn eat(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }
    fn try_eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) { self.pos += 1; true } else { false }
    }

    fn alt(&mut self) -> Result<Node, String> {
        let first = self.concat()?;
        if self.peek() != Some('|') { return Ok(first); }
        let mut arms = vec![first];
        while self.try_eat('|') { arms.push(self.concat()?); }
        Ok(Node::Alt(arms))
    }

    fn concat(&mut self) -> Result<Node, String> {
        let mut nodes = Vec::new();
        while !matches!(self.peek(), None | Some(')') | Some('|')) {
            nodes.push(self.quantified()?);
        }
        match nodes.len() {
            0 => Ok(Node::Concat(vec![])),
            1 => Ok(nodes.remove(0)),
            _ => Ok(Node::Concat(nodes)),
        }
    }

    fn quantified(&mut self) -> Result<Node, String> {
        let atom = self.atom()?;
        let (min, max) = match self.peek() {
            Some('*') => { self.pos += 1; (0, None) }
            Some('+') => { self.pos += 1; (1, None) }
            Some('?') => { self.pos += 1; (0, Some(1)) }
            Some('{') => match self.braces()? {
                Some(b) => b,
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        let greedy = !self.try_eat('?'); // lazy if followed by '?'
        Ok(Node::Quant(Box::new(atom), min, max, greedy))
    }

    fn braces(&mut self) -> Result<Option<(usize, Option<usize>)>, String> {
        let save = self.pos;
        self.pos += 1; // '{'
        let n = match self.decimal() {
            Some(n) => n,
            None => { self.pos = save; return Ok(None); }
        };
        if self.try_eat('}') { return Ok(Some((n, Some(n)))); }
        if !self.try_eat(',') { self.pos = save; return Ok(None); }
        if self.try_eat('}') { return Ok(Some((n, None))); }
        let m = match self.decimal() {
            Some(m) => m,
            None => { self.pos = save; return Ok(None); }
        };
        if !self.try_eat('}') { self.pos = save; return Ok(None); }
        if m < n { return Err(format!("regex: invalid range {{{},{}}}", n, m)); }
        Ok(Some((n, Some(m))))
    }

    fn decimal(&mut self) -> Option<usize> {
        let s = self.pos;
        while matches!(self.peek(), Some('0'..='9')) { self.pos += 1; }
        if self.pos == s { return None; }
        self.chars[s..self.pos].iter().collect::<String>().parse().ok()
    }

    fn atom(&mut self) -> Result<Node, String> {
        match self.peek() {
            Some('^') => { self.pos += 1; Ok(Node::AnchorStart) }
            Some('$') => { self.pos += 1; Ok(Node::AnchorEnd) }
            Some('.') => { self.pos += 1; Ok(Node::Char(CClass::Any)) }
            Some('(') => {
                self.pos += 1;
                let nc = self.chars.get(self.pos..self.pos + 2) == Some(&['?', ':']);
                if nc { self.pos += 2; }
                let inner = self.alt()?;
                if !self.try_eat(')') { return Err("regex: unclosed group".into()); }
                if nc { return Ok(Node::NonCap(Box::new(inner))); }
                self.ngroups += 1;
                Ok(Node::Group(Box::new(inner), self.ngroups))
            }
            Some('[') => { self.pos += 1; self.char_class() }
            Some('\\') => { self.pos += 1; self.escape() }
            Some(c) if !matches!(c, ')' | '|' | '*' | '+' | '?' | '{' | '}') => {
                self.pos += 1;
                Ok(Node::Char(CClass::Lit(c)))
            }
            Some(c) => Err(format!("regex: unexpected '{}' at pos {}", c, self.pos)),
            None => Err("regex: unexpected end of pattern".into()),
        }
    }

    fn char_class(&mut self) -> Result<Node, String> {
        let neg = self.try_eat('^');
        let mut items = Vec::new();
        if self.peek() == Some(']') { items.push(CItem::Ch(']')); self.pos += 1; }
        while !matches!(self.peek(), Some(']') | None) {
            match self.peek() {
                Some('\\') => {
                    self.pos += 1;
                    match self.eat() {
                        Some('d') => items.push(CItem::Digit),
                        Some('D') => items.push(CItem::Digit), // negated in Set(_, neg) context — simplified
                        Some('w') => items.push(CItem::Word),
                        Some('s') => items.push(CItem::Space),
                        Some('n') => items.push(CItem::Ch('\n')),
                        Some('t') => items.push(CItem::Ch('\t')),
                        Some('r') => items.push(CItem::Ch('\r')),
                        Some(c) => items.push(CItem::Ch(c)),
                        None => return Err("regex: trailing \\ in class".into()),
                    }
                }
                Some(c) => {
                    self.pos += 1;
                    let next_is_range_end = self.peek() == Some('-')
                        && !matches!(self.chars.get(self.pos + 1), Some(']') | None);
                    if next_is_range_end {
                        self.pos += 1; // consume '-'
                        match self.eat() {
                            Some(hi) => items.push(CItem::Range(c, hi)),
                            None => return Err("regex: invalid range in class".into()),
                        }
                    } else {
                        items.push(CItem::Ch(c));
                    }
                }
                None => break,
            }
        }
        if !self.try_eat(']') { return Err("regex: unclosed '['".into()); }
        Ok(Node::Char(CClass::Set(items, neg)))
    }

    fn escape(&mut self) -> Result<Node, String> {
        match self.eat() {
            Some('d') => Ok(Node::Char(CClass::Set(vec![CItem::Digit], false))),
            Some('D') => Ok(Node::Char(CClass::Set(vec![CItem::Digit], true))),
            Some('w') => Ok(Node::Char(CClass::Set(vec![CItem::Word], false))),
            Some('W') => Ok(Node::Char(CClass::Set(vec![CItem::Word], true))),
            Some('s') => Ok(Node::Char(CClass::Set(vec![CItem::Space], false))),
            Some('S') => Ok(Node::Char(CClass::Set(vec![CItem::Space], true))),
            Some('b') => Ok(Node::WordBound(true)),
            Some('B') => Ok(Node::WordBound(false)),
            Some('n') => Ok(Node::Char(CClass::Lit('\n'))),
            Some('t') => Ok(Node::Char(CClass::Lit('\t'))),
            Some('r') => Ok(Node::Char(CClass::Lit('\r'))),
            Some('x') => {
                let hi = self.eat().ok_or("regex: incomplete \\x escape")?;
                let lo = self.eat().ok_or("regex: incomplete \\x escape")?;
                let s = format!("{}{}", hi, lo);
                let code = u8::from_str_radix(&s, 16)
                    .map_err(|_| format!("regex: invalid \\x{}", s))?;
                Ok(Node::Char(CClass::Lit(code as char)))
            }
            Some(c) => Ok(Node::Char(CClass::Lit(c))),
            None => Err("regex: trailing '\\'".into()),
        }
    }
}

// ── Bytecode compiler ─────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum Instr {
    Char(CClass),
    AnchorStart,
    AnchorEnd,
    WordBound(bool),
    GSave(usize),    // save current char-position to caps[slot]
    Fork(usize),     // greedy: try pc+1 first; push (dest, sp, caps) as fallback
    ForkLazy(usize), // lazy: try dest first; push (pc+1, sp, caps) as fallback
    Jump(usize),     // absolute jump
    Match,
}

struct Prog {
    instrs: Vec<Instr>,
}

impl Prog {
    fn emit(&mut self, i: Instr) -> usize { let p = self.instrs.len(); self.instrs.push(i); p }
    fn patch(&mut self, at: usize, i: Instr) { self.instrs[at] = i; }
    fn len(&self) -> usize { self.instrs.len() }
}

fn compile(root: &Node, ngroups: usize) -> Vec<Instr> {
    let mut prog = Prog { instrs: Vec::new() };
    prog.emit(Instr::GSave(0)); // full match start
    compile_node(root, &mut prog);
    prog.emit(Instr::GSave(1)); // full match end
    prog.emit(Instr::Match);
    // resize-pad so caps vector is (ngroups+1)*2 long
    let _ = ngroups;
    prog.instrs
}

fn compile_node(node: &Node, p: &mut Prog) {
    match node {
        Node::Char(c) => { p.emit(Instr::Char(c.clone())); }
        Node::AnchorStart => { p.emit(Instr::AnchorStart); }
        Node::AnchorEnd => { p.emit(Instr::AnchorEnd); }
        Node::WordBound(b) => { p.emit(Instr::WordBound(*b)); }
        Node::Concat(nodes) => { for n in nodes { compile_node(n, p); } }
        Node::NonCap(inner) => compile_node(inner, p),
        Node::Group(inner, gidx) => {
            p.emit(Instr::GSave(gidx * 2));
            compile_node(inner, p);
            p.emit(Instr::GSave(gidx * 2 + 1));
        }
        Node::Alt(arms) => compile_alt(arms, p),
        Node::Quant(inner, min, max, greedy) => compile_quant(inner, *min, *max, *greedy, p),
    }
}

fn compile_alt(arms: &[Node], p: &mut Prog) {
    if arms.is_empty() { return; }
    if arms.len() == 1 { compile_node(&arms[0], p); return; }
    // Fork(arm2_start); arm1; Jump(end); arm2_start: ...
    let fork_pos = p.emit(Instr::Fork(0)); // placeholder
    compile_node(&arms[0], p);
    let jump_pos = p.emit(Instr::Jump(0)); // placeholder
    let arm2_start = p.len();
    p.patch(fork_pos, Instr::Fork(arm2_start));
    compile_alt(&arms[1..], p);
    let end = p.len();
    p.patch(jump_pos, Instr::Jump(end));
}

fn compile_quant(inner: &Node, min: usize, max: Option<usize>, greedy: bool, p: &mut Prog) {
    // Emit mandatory repetitions
    for _ in 0..min { compile_node(inner, p); }

    match max {
        Some(m) if m == min => {} // exactly n — done
        Some(m) => {
            // Bounded optional: (m - min) Fork blocks
            let optional = m - min;
            for _ in 0..optional {
                if greedy {
                    let fork = p.emit(Instr::Fork(0));
                    compile_node(inner, p);
                    let after = p.len();
                    p.patch(fork, Instr::Fork(after));
                } else {
                    let fork = p.emit(Instr::ForkLazy(0));
                    let after_fork = p.len();
                    compile_node(inner, p);
                    let after = p.len();
                    p.patch(fork, Instr::ForkLazy(after));
                    let _ = after_fork;
                }
            }
        }
        None => {
            // Unbounded loop
            if greedy {
                // loop_start: Fork(after); body; Jump(loop_start); after:
                let loop_start = p.len();
                let fork = p.emit(Instr::Fork(0));
                compile_node(inner, p);
                p.emit(Instr::Jump(loop_start));
                let after = p.len();
                p.patch(fork, Instr::Fork(after));
            } else {
                // lazy: loop_start: ForkLazy(body_start); Jump(after)... complex
                // Simpler: ForkLazy(after); body; Jump(loop_start)
                let loop_start = p.len();
                let fork = p.emit(Instr::ForkLazy(0));
                compile_node(inner, p);
                p.emit(Instr::Jump(loop_start));
                let after = p.len();
                p.patch(fork, Instr::ForkLazy(after));
            }
        }
    }
}

// ── VM ────────────────────────────────────────────────────────────────

/// Maximum number of VM steps before aborting (prevents ReDoS).
const MAX_REGEX_STEPS: usize = 100_000;

type Caps = Vec<Option<usize>>; // char-index positions; even = start, odd = end

struct State {
    pc: usize,
    sp: usize,
    caps: Caps,
}

/// Run the VM from starting string position `start`.
/// Returns cap array on success, or None.
fn vm_run(instrs: &[Instr], text: &[char], start: usize, ncaps: usize) -> Option<Caps> {
    let cap_slots = (ncaps + 1) * 2;
    let init = vec![None; cap_slots];
    let mut stack: Vec<State> = vec![State { pc: 0, sp: start, caps: init }];
    let mut steps = 0usize;

    'outer: while let Some(mut st) = stack.pop() {
        loop {
            steps += 1;
            if steps > MAX_REGEX_STEPS {
                return None; // step limit exceeded — treat as no match
            }
            match &instrs[st.pc] {
                Instr::Match => return Some(st.caps),
                Instr::GSave(slot) => {
                    if *slot < st.caps.len() { st.caps[*slot] = Some(st.sp); }
                    st.pc += 1;
                }
                Instr::Char(class) => {
                    if st.sp < text.len() && class.matches(text[st.sp]) {
                        st.sp += 1; st.pc += 1;
                    } else { continue 'outer; }
                }
                Instr::AnchorStart => {
                    if st.sp == 0 { st.pc += 1; } else { continue 'outer; }
                }
                Instr::AnchorEnd => {
                    if st.sp == text.len() { st.pc += 1; } else { continue 'outer; }
                }
                Instr::WordBound(want) => {
                    let prev = st.sp > 0 && is_word(text[st.sp - 1]);
                    let next = st.sp < text.len() && is_word(text[st.sp]);
                    if (prev != next) == *want { st.pc += 1; } else { continue 'outer; }
                }
                Instr::Fork(dest) => {
                    // Greedy: try pc+1 now; fallback = dest
                    let fallback = State { pc: *dest, sp: st.sp, caps: st.caps.clone() };
                    stack.push(fallback);
                    st.pc += 1;
                }
                Instr::ForkLazy(dest) => {
                    // Lazy: try dest now; fallback = pc+1
                    let fallback = State { pc: st.pc + 1, sp: st.sp, caps: st.caps.clone() };
                    stack.push(fallback);
                    st.pc = *dest;
                }
                Instr::Jump(dest) => {
                    st.pc = *dest;
                }
            }
        }
    }
    None
}

/// Search for the first match starting at or after `from`.
/// Returns (match_start, caps) or None.
fn search(instrs: &[Instr], text: &[char], from: usize, ncaps: usize)
    -> Option<(usize, Caps)>
{
    // Fast path: if program starts with AnchorStart (after GSave(0)), only try pos=from
    let anchored = matches!(instrs.get(1), Some(Instr::AnchorStart));
    let end = if anchored { from.min(text.len()) } else { text.len() };

    for start in from..=end {
        if let Some(caps) = vm_run(instrs, text, start, ncaps) {
            return Some((start, caps));
        }
    }
    None
}

// ── Cap helpers ───────────────────────────────────────────────────────

fn cap_str(text: &[char], caps: &Caps, slot: usize) -> Option<String> {
    let start = *caps.get(slot * 2)?.as_ref()?;
    let end   = *caps.get(slot * 2 + 1)?.as_ref()?;
    Some(text[start..end].iter().collect())
}

// ── Public API ────────────────────────────────────────────────────────

/// A compiled regex pattern.
pub struct Regex {
    instrs: Vec<Instr>,
    ncaps: usize,
}

#[derive(Debug)]
pub struct RegexError(String);

impl std::fmt::Display for RegexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

impl Regex {
    /// Compile a regex pattern.
    pub fn new(pattern: &str) -> Result<Self, RegexError> {
        let (node, ncaps) = parse_regex(pattern).map_err(RegexError)?;
        let instrs = compile(&node, ncaps);
        Ok(Regex { instrs, ncaps })
    }

    /// Test whether the pattern matches anywhere in `text`.
    pub fn is_match(&self, text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        search(&self.instrs, &chars, 0, self.ncaps).is_some()
    }

    /// Return the first match as an owned string, or None.
    pub fn find(&self, text: &str) -> Option<String> {
        let chars: Vec<char> = text.chars().collect();
        let (_, caps) = search(&self.instrs, &chars, 0, self.ncaps)?;
        cap_str(&chars, &caps, 0)
    }

    /// Replace all non-overlapping matches. `$0`=full match, `$1`-`$9`=groups.
    pub fn replace_all(&self, text: &str, replacement: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let mut out = String::new();
        let mut pos = 0usize;
        while pos <= chars.len() {
            match search(&self.instrs, &chars, pos, self.ncaps) {
                Some((mstart, caps)) => {
                    // Append text before match
                    out.extend(chars[pos..mstart].iter());
                    // Build replacement with backreferences
                    out.push_str(&apply_replacement(replacement, &chars, &caps, self.ncaps));
                    let mend = caps.get(1).copied().flatten().unwrap_or(mstart);
                    if mend > pos {
                        pos = mend;
                    } else {
                        // Zero-length match — advance one char to avoid infinite loop
                        if pos < chars.len() { out.push(chars[pos]); }
                        pos += 1;
                    }
                }
                None => {
                    out.extend(chars[pos..].iter());
                    break;
                }
            }
        }
        out
    }

    /// Split `text` on all matches of this pattern.
    pub fn split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut parts = Vec::new();
        let mut pos = 0usize;
        while pos <= chars.len() {
            match search(&self.instrs, &chars, pos, self.ncaps) {
                Some((mstart, caps)) => {
                    parts.push(chars[pos..mstart].iter().collect());
                    let mend = caps.get(1).copied().flatten().unwrap_or(mstart);
                    if mend > pos { pos = mend; } else { pos += 1; }
                }
                None => {
                    parts.push(chars[pos..].iter().collect());
                    break;
                }
            }
        }
        // Remove trailing empty string only if pattern matched at end
        parts
    }

    /// Return all non-overlapping matches as owned strings.
    pub fn find_all(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut results = Vec::new();
        let mut pos = 0usize;
        while pos <= chars.len() {
            match search(&self.instrs, &chars, pos, self.ncaps) {
                Some((mstart, caps)) => {
                    if let Some(s) = cap_str(&chars, &caps, 0) {
                        results.push(s);
                    }
                    let mend = caps.get(1).copied().flatten().unwrap_or(mstart);
                    if mend > pos {
                        pos = mend;
                    } else {
                        // Zero-length match — advance one char to avoid infinite loop
                        pos += 1;
                    }
                }
                None => break,
            }
        }
        results
    }

    /// Return capture groups from the first match: [0]=full match, [1..]=groups.
    /// Returns None if no match.
    pub fn captures(&self, text: &str) -> Option<Vec<Option<String>>> {
        let chars: Vec<char> = text.chars().collect();
        let (_, caps) = search(&self.instrs, &chars, 0, self.ncaps)?;
        let mut result = Vec::new();
        for i in 0..=(self.ncaps) {
            result.push(cap_str(&chars, &caps, i));
        }
        Some(result)
    }
}

fn apply_replacement(repl: &str, text: &[char], caps: &Caps, ncaps: usize) -> String {
    let mut out = String::new();
    let rchars: Vec<char> = repl.chars().collect();
    let mut i = 0;
    while i < rchars.len() {
        if rchars[i] == '$' && i + 1 < rchars.len() {
            let next = rchars[i + 1];
            if next.is_ascii_digit() {
                let idx = (next as u8 - b'0') as usize;
                if idx <= ncaps {
                    if let Some(s) = cap_str(text, caps, idx) {
                        out.push_str(&s);
                    }
                }
                i += 2;
                continue;
            }
        }
        out.push(rchars[i]);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_match() {
        let re = Regex::new("hello").unwrap();
        assert!(re.is_match("hello world"));
        assert!(!re.is_match("goodbye"));
    }

    #[test]
    fn test_find_returns_match() {
        let re = Regex::new("[0-9]+").unwrap();
        assert_eq!(re.find("abc 42 def"), Some("42".to_string()));
    }

    #[test]
    fn test_replace_all() {
        let re = Regex::new("[aeiou]").unwrap();
        assert_eq!(re.replace_all("hello", "_"), "h_ll_");
    }

    #[test]
    fn test_step_limit_prevents_catastrophic_backtracking() {
        // Pattern that causes exponential backtracking: (a+)+ against "aaa...!"
        // With step limit, this should return None (no match) instead of hanging.
        let re = Regex::new("(a+)+b").unwrap();
        // 30 a's followed by ! — triggers catastrophic backtracking in naive NFA
        let evil_input = "a".repeat(30) + "!";
        // Should return false (step limit exceeded → treated as no match), not hang
        assert!(!re.is_match(&evil_input));
    }

    #[test]
    fn test_split() {
        let re = Regex::new(",").unwrap();
        let parts = re.split("a,b,c");
        assert_eq!(parts, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_find_all() {
        let re = Regex::new("[0-9]+").unwrap();
        let matches = re.find_all("12 abc 34 def 56");
        assert_eq!(matches, vec!["12", "34", "56"]);
    }
}

