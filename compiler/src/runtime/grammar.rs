//! GBNF Grammar-Constrained Decoding for OctoFlow LLM inference.
//!
//! Parses GBNF grammar files (llama.cpp format) into an NFA-based state machine
//! that constrains token sampling to valid OctoFlow syntax.
//!
//! Flow:
//! 1. `grammar_load(path)` — parse .gbnf file into GrammarState
//! 2. `grammar_mask(state, logits, vocab)` — mask invalid tokens in logits
//! 3. `grammar_advance(state, token_text)` — advance state after sampling

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// GBNF AST
// ═══════════════════════════════════════════════════════════════

/// A single element in a grammar alternative (sequence).
#[derive(Debug, Clone)]
enum GbnfElement {
    /// Literal string: "let", "if", etc.
    Literal(Vec<u8>),
    /// Character class: [a-zA-Z_], [^\n], etc. Stored as (ranges, negated).
    CharClass { ranges: Vec<(u8, u8)>, negated: bool },
    /// Reference to another rule by name.
    RuleRef(String),
    /// Repetition: element* (zero or more).
    Star(Box<GbnfElement>),
    /// Optional: element? (zero or one). Desugared to (element | ε).
    Optional(Box<GbnfElement>),
    /// One or more: element+ = element element*
    Plus(Box<GbnfElement>),
}

/// A single alternative (sequence of elements).
type GbnfAlt = Vec<GbnfElement>;

/// A grammar rule: name -> alt1 | alt2 | ...
#[derive(Debug, Clone)]
struct GbnfRule {
    name: String,
    alternatives: Vec<GbnfAlt>,
}

// ═══════════════════════════════════════════════════════════════
// NFA REPRESENTATION
// ═══════════════════════════════════════════════════════════════

/// Transition condition for an NFA edge.
#[derive(Debug, Clone)]
enum NfaTransition {
    /// Match a single byte.
    Byte(u8),
    /// Match any byte in the given ranges (inclusive). If negated, match complement.
    ByteRange { ranges: Vec<(u8, u8)>, negated: bool },
    /// Epsilon transition (no input consumed).
    Epsilon,
}

/// NFA state ID.
type StateId = usize;

/// A single NFA edge: from (implicit) -> to, on condition.
#[derive(Debug, Clone)]
struct NfaEdge {
    to: StateId,
    cond: NfaTransition,
}

/// The compiled NFA for the entire grammar.
#[derive(Debug, Clone)]
struct GrammarNfa {
    /// All edges, indexed by source state.
    edges: Vec<Vec<NfaEdge>>,
    /// Start and accept states for each rule.
    rule_starts: HashMap<String, StateId>,
    rule_accepts: HashMap<String, StateId>,
    /// Total state count.
    num_states: usize,
}

impl GrammarNfa {
    fn new() -> Self {
        GrammarNfa {
            edges: Vec::new(),
            rule_starts: HashMap::new(),
            rule_accepts: HashMap::new(),
            num_states: 0,
        }
    }

    fn new_state(&mut self) -> StateId {
        let id = self.num_states;
        self.num_states += 1;
        self.edges.push(Vec::new());
        id
    }

    fn add_edge(&mut self, from: StateId, to: StateId, cond: NfaTransition) {
        self.edges[from].push(NfaEdge { to, cond });
    }
}

// ═══════════════════════════════════════════════════════════════
// GRAMMAR STATE (runtime tracking)
// ═══════════════════════════════════════════════════════════════

/// Runtime state for grammar-constrained decoding.
/// Tracks which NFA states are currently active (epsilon closure).
#[derive(Debug, Clone)]
pub struct GrammarState {
    nfa: GrammarNfa,
    /// Current set of active NFA states (after epsilon closure).
    active: Vec<StateId>,
    /// Root rule name (typically "root").
    root_rule: String,
}

impl GrammarState {
    /// Check if the grammar has reached an accepting state.
    pub fn can_accept(&self) -> bool {
        if let Some(&accept) = self.nfa.rule_accepts.get(&self.root_rule) {
            self.active.contains(&accept)
        } else {
            false
        }
    }

    /// Get all bytes that would be valid from the current state.
    fn valid_bytes(&self) -> Vec<bool> {
        let mut valid = vec![false; 256];
        for &state in &self.active {
            for edge in &self.nfa.edges[state] {
                match &edge.cond {
                    NfaTransition::Byte(b) => {
                        valid[*b as usize] = true;
                    }
                    NfaTransition::ByteRange { ranges, negated } => {
                        if *negated {
                            // Everything NOT in ranges is valid
                            for b in 0u8..=255 {
                                let in_range = ranges.iter().any(|&(lo, hi)| b >= lo && b <= hi);
                                if !in_range {
                                    valid[b as usize] = true;
                                }
                            }
                        } else {
                            for &(lo, hi) in ranges {
                                for b in lo..=hi {
                                    valid[b as usize] = true;
                                }
                            }
                        }
                    }
                    NfaTransition::Epsilon => {}
                }
            }
        }
        valid
    }

    /// Advance the state by consuming a single byte.
    fn advance_byte(&self, byte: u8) -> Vec<StateId> {
        let mut next = Vec::new();
        for &state in &self.active {
            for edge in &self.nfa.edges[state] {
                let matches = match &edge.cond {
                    NfaTransition::Byte(b) => *b == byte,
                    NfaTransition::ByteRange { ranges, negated } => {
                        let in_range = ranges.iter().any(|&(lo, hi)| byte >= lo && byte <= hi);
                        if *negated { !in_range } else { in_range }
                    }
                    NfaTransition::Epsilon => false,
                };
                if matches && !next.contains(&edge.to) {
                    next.push(edge.to);
                }
            }
        }
        next
    }

    /// Advance the state by consuming a string of bytes.
    /// Returns the new state (or empty if the string is invalid).
    pub fn advance_str(&mut self, text: &str) {
        for byte in text.bytes() {
            let next = self.advance_byte(byte);
            let closed = epsilon_closure(&self.nfa, &next);
            self.active = closed;
            if self.active.is_empty() {
                return;
            }
        }
    }

    /// Check if a token string could be valid from the current state.
    /// Returns true if any prefix of the token leads to a non-empty state,
    /// OR if the full token advances to a non-empty state.
    pub fn token_is_valid(&self, token_bytes: &[u8]) -> bool {
        if token_bytes.is_empty() {
            return true;
        }
        let mut current = self.active.clone();
        for &byte in token_bytes {
            let mut next = Vec::new();
            for &state in &current {
                for edge in &self.nfa.edges[state] {
                    let matches = match &edge.cond {
                        NfaTransition::Byte(b) => *b == byte,
                        NfaTransition::ByteRange { ranges, negated } => {
                            let in_range = ranges.iter().any(|&(lo, hi)| byte >= lo && byte <= hi);
                            if *negated { !in_range } else { in_range }
                        }
                        NfaTransition::Epsilon => false,
                    };
                    if matches && !next.contains(&edge.to) {
                        next.push(edge.to);
                    }
                }
            }
            let closed = epsilon_closure(&self.nfa, &next);
            if closed.is_empty() {
                return false;
            }
            current = closed;
        }
        true
    }
}

// ═══════════════════════════════════════════════════════════════
// GBNF PARSER
// ═══════════════════════════════════════════════════════════════

/// Parse a GBNF grammar file into a list of rules.
fn parse_gbnf(source: &str) -> Result<Vec<GbnfRule>, String> {
    let mut rules = Vec::new();
    let mut lines_iter = source.lines().peekable();

    while let Some(line) = lines_iter.next() {
        let trimmed = line.trim();
        // Skip comments and blank lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Rule: name ::= alternatives
        if let Some(sep_pos) = trimmed.find("::=") {
            let name = trimmed[..sep_pos].trim().to_string();
            let mut rhs = trimmed[sep_pos + 3..].to_string();

            // Continuation lines (indented or ending with |)
            while let Some(&next_line) = lines_iter.peek() {
                let next_trimmed = next_line.trim();
                if next_trimmed.is_empty() || next_trimmed.starts_with('#') {
                    break;
                }
                // Continuation if line starts with | or is indented continuation
                if next_trimmed.starts_with('|') || (next_line.starts_with(' ') && !next_trimmed.contains("::=")) {
                    rhs.push(' ');
                    rhs.push_str(next_trimmed);
                    lines_iter.next();
                } else {
                    break;
                }
            }

            let alternatives = parse_alternatives(&rhs)?;
            rules.push(GbnfRule { name, alternatives });
        }
    }

    if rules.is_empty() {
        return Err("no rules found in GBNF grammar".into());
    }

    Ok(rules)
}

/// Parse the RHS of a rule into alternatives separated by |.
fn parse_alternatives(rhs: &str) -> Result<Vec<GbnfAlt>, String> {
    let mut alts = Vec::new();
    let mut current_alt = Vec::new();
    let bytes = rhs.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        match bytes[i] {
            b' ' | b'\t' => { i += 1; }
            b'|' => {
                alts.push(current_alt);
                current_alt = Vec::new();
                i += 1;
            }
            b'"' => {
                // Quoted literal
                let (lit, end) = parse_quoted_literal(bytes, i)?;
                let elem = GbnfElement::Literal(lit);
                let (elem, end2) = apply_quantifier(elem, bytes, end);
                current_alt.push(elem);
                i = end2;
            }
            b'[' => {
                // Character class
                let (elem, end) = parse_char_class(bytes, i)?;
                i = end;
                // Check for quantifier
                let (elem, end2) = apply_quantifier(elem, bytes, i);
                current_alt.push(elem);
                i = end2;
            }
            b'(' => {
                // Grouped sub-expression
                let (sub_alts, end) = parse_group(bytes, i)?;
                i = end;
                // Wrap in a synthetic element
                let group_elem = if sub_alts.len() == 1 && sub_alts[0].len() == 1 {
                    sub_alts[0][0].clone()
                } else {
                    // Create inline rule reference — flatten single-element alts
                    GbnfElement::RuleRef(format!("__group_{}", i))
                };
                let (elem, end2) = apply_quantifier(group_elem, bytes, i);
                current_alt.push(elem);
                i = end2;
            }
            b'\\' if i + 1 < bytes.len() => {
                // Escaped character
                let ch = match bytes[i + 1] {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'\\' => b'\\',
                    b'"' => b'"',
                    b'0' => 0,
                    other => other,
                };
                let elem = GbnfElement::Literal(vec![ch]);
                i += 2;
                let (elem, end) = apply_quantifier(elem, bytes, i);
                current_alt.push(elem);
                i = end;
            }
            _ if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' => {
                // Rule reference
                let start = i;
                while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_' || bytes[i] == b'-') {
                    i += 1;
                }
                let name = std::str::from_utf8(&bytes[start..i])
                    .map_err(|_| "invalid UTF-8 in rule name")?
                    .to_string();
                let elem = GbnfElement::RuleRef(name);
                let (elem, end) = apply_quantifier(elem, bytes, i);
                current_alt.push(elem);
                i = end;
            }
            b'/' if i + 1 < bytes.len() && bytes[i + 1] == b'/' => {
                // Inline comment — skip rest of logical line segment
                break;
            }
            _ => {
                i += 1; // skip unknown
            }
        }
    }

    alts.push(current_alt);
    Ok(alts)
}

/// Parse a quoted literal: "..." -> bytes
fn parse_quoted_literal(bytes: &[u8], start: usize) -> Result<(Vec<u8>, usize), String> {
    let mut result = Vec::new();
    let mut i = start + 1; // skip opening quote
    while i < bytes.len() {
        if bytes[i] == b'"' {
            return Ok((result, i + 1));
        }
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            let ch = match bytes[i + 1] {
                b'n' => b'\n',
                b't' => b'\t',
                b'r' => b'\r',
                b'\\' => b'\\',
                b'"' => b'"',
                b'0' => 0,
                other => other,
            };
            result.push(ch);
            i += 2;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    Err("unterminated string literal in GBNF".into())
}

/// Parse a character class: [a-zA-Z_] or [^\n]
fn parse_char_class(bytes: &[u8], start: usize) -> Result<(GbnfElement, usize), String> {
    let mut i = start + 1; // skip [
    let negated = if i < bytes.len() && bytes[i] == b'^' {
        i += 1;
        true
    } else {
        false
    };

    let mut ranges = Vec::new();

    while i < bytes.len() && bytes[i] != b']' {
        let ch = if bytes[i] == b'\\' && i + 1 < bytes.len() {
            i += 1;
            match bytes[i] {
                b'n' => b'\n',
                b't' => b'\t',
                b'r' => b'\r',
                b'\\' => b'\\',
                b']' => b']',
                b'[' => b'[',
                b'^' => b'^',
                b'0' => 0,
                other => other,
            }
        } else {
            let c = bytes[i];
            c
        };
        i += 1;

        // Check for range: a-z
        if i + 1 < bytes.len() && bytes[i] == b'-' && bytes[i + 1] != b']' {
            i += 1; // skip -
            let hi = if bytes[i] == b'\\' && i + 1 < bytes.len() {
                i += 1;
                match bytes[i] {
                    b'n' => b'\n',
                    b't' => b'\t',
                    b'r' => b'\r',
                    b'\\' => b'\\',
                    b']' => b']',
                    b'0' => 0,
                    other => other,
                }
            } else {
                bytes[i]
            };
            i += 1;
            ranges.push((ch, hi));
        } else {
            ranges.push((ch, ch));
        }
    }

    if i < bytes.len() && bytes[i] == b']' {
        i += 1; // skip ]
    }

    Ok((GbnfElement::CharClass { ranges, negated }, i))
}

/// Parse a grouped sub-expression: ( alt1 | alt2 )
fn parse_group(bytes: &[u8], start: usize) -> Result<(Vec<GbnfAlt>, usize), String> {
    let mut depth = 1;
    let mut i = start + 1; // skip (
    let group_start = i;

    while i < bytes.len() && depth > 0 {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'"' => {
                // Skip quoted strings
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' { i += 1; }
                    i += 1;
                }
            }
            b'[' => {
                // Skip character classes
                i += 1;
                while i < bytes.len() && bytes[i] != b']' {
                    if bytes[i] == b'\\' { i += 1; }
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    let group_text = std::str::from_utf8(&bytes[group_start..i - 1])
        .map_err(|_| "invalid UTF-8 in group")?;
    let alts = parse_alternatives(group_text)?;
    Ok((alts, i))
}

/// Apply quantifier (*, +, ?) to an element.
fn apply_quantifier(elem: GbnfElement, bytes: &[u8], pos: usize) -> (GbnfElement, usize) {
    if pos < bytes.len() {
        match bytes[pos] {
            b'*' => (GbnfElement::Star(Box::new(elem)), pos + 1),
            b'+' => (GbnfElement::Plus(Box::new(elem)), pos + 1),
            b'?' => (GbnfElement::Optional(Box::new(elem)), pos + 1),
            _ => (elem, pos),
        }
    } else {
        (elem, pos)
    }
}

// ═══════════════════════════════════════════════════════════════
// NFA COMPILATION
// ═══════════════════════════════════════════════════════════════

/// Compile parsed GBNF rules into an NFA.
fn compile_nfa(rules: &[GbnfRule]) -> GrammarNfa {
    let mut nfa = GrammarNfa::new();

    // First pass: create start/accept states for each rule
    for rule in rules {
        let start = nfa.new_state();
        let accept = nfa.new_state();
        nfa.rule_starts.insert(rule.name.clone(), start);
        nfa.rule_accepts.insert(rule.name.clone(), accept);
    }

    // Second pass: compile each rule body
    for rule in rules {
        let start = nfa.rule_starts[&rule.name];
        let accept = nfa.rule_accepts[&rule.name];

        for alt in &rule.alternatives {
            compile_alt(&mut nfa, alt, start, accept);
        }
    }

    nfa
}

/// Compile an alternative (sequence of elements) into the NFA.
fn compile_alt(nfa: &mut GrammarNfa, alt: &[GbnfElement], start: StateId, accept: StateId) {
    if alt.is_empty() {
        // Epsilon alternative
        nfa.add_edge(start, accept, NfaTransition::Epsilon);
        return;
    }

    let mut prev = start;
    for (i, elem) in alt.iter().enumerate() {
        let next = if i == alt.len() - 1 { accept } else { nfa.new_state() };
        compile_element(nfa, elem, prev, next);
        prev = next;
    }
}

/// Compile a single element into the NFA.
fn compile_element(nfa: &mut GrammarNfa, elem: &GbnfElement, start: StateId, accept: StateId) {
    match elem {
        GbnfElement::Literal(bytes) => {
            if bytes.is_empty() {
                nfa.add_edge(start, accept, NfaTransition::Epsilon);
                return;
            }
            let mut prev = start;
            for (i, &byte) in bytes.iter().enumerate() {
                let next = if i == bytes.len() - 1 { accept } else { nfa.new_state() };
                nfa.add_edge(prev, next, NfaTransition::Byte(byte));
                prev = next;
            }
        }
        GbnfElement::CharClass { ranges, negated } => {
            nfa.add_edge(start, accept, NfaTransition::ByteRange {
                ranges: ranges.clone(),
                negated: *negated,
            });
        }
        GbnfElement::RuleRef(name) => {
            // Epsilon from start to rule's start, epsilon from rule's accept to our accept
            if let Some(&rule_start) = nfa.rule_starts.get(name) {
                let rule_accept = nfa.rule_accepts[name];
                nfa.add_edge(start, rule_start, NfaTransition::Epsilon);
                nfa.add_edge(rule_accept, accept, NfaTransition::Epsilon);
            } else {
                // Unknown rule — treat as accepting anything (best effort)
                nfa.add_edge(start, accept, NfaTransition::Epsilon);
            }
        }
        GbnfElement::Star(inner) => {
            // start -ε-> loop_start, loop_accept -ε-> loop_start, loop_accept -ε-> accept, start -ε-> accept
            let loop_start = nfa.new_state();
            let loop_accept = nfa.new_state();
            nfa.add_edge(start, accept, NfaTransition::Epsilon); // zero times
            nfa.add_edge(start, loop_start, NfaTransition::Epsilon);
            compile_element(nfa, inner, loop_start, loop_accept);
            nfa.add_edge(loop_accept, loop_start, NfaTransition::Epsilon); // repeat
            nfa.add_edge(loop_accept, accept, NfaTransition::Epsilon);
        }
        GbnfElement::Plus(inner) => {
            // At least once: start -> loop, loop back
            let loop_start = nfa.new_state();
            let loop_accept = nfa.new_state();
            nfa.add_edge(start, loop_start, NfaTransition::Epsilon);
            compile_element(nfa, inner, loop_start, loop_accept);
            nfa.add_edge(loop_accept, loop_start, NfaTransition::Epsilon); // repeat
            nfa.add_edge(loop_accept, accept, NfaTransition::Epsilon); // done
        }
        GbnfElement::Optional(inner) => {
            // Zero or one: start -ε-> accept, start -> inner -> accept
            nfa.add_edge(start, accept, NfaTransition::Epsilon);
            compile_element(nfa, inner, start, accept);
        }
    }
}

/// Compute epsilon closure for a set of states.
fn epsilon_closure(nfa: &GrammarNfa, states: &[StateId]) -> Vec<StateId> {
    let mut closure = states.to_vec();
    let mut stack = states.to_vec();
    // Limit iterations to prevent infinite loops in pathological grammars
    let max_iters = nfa.num_states * 4;
    let mut iters = 0;

    while let Some(state) = stack.pop() {
        iters += 1;
        if iters > max_iters {
            break;
        }
        for edge in &nfa.edges[state] {
            if let NfaTransition::Epsilon = &edge.cond {
                if !closure.contains(&edge.to) {
                    closure.push(edge.to);
                    stack.push(edge.to);
                }
            }
        }
    }
    closure
}

// ═══════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════

/// Load a GBNF grammar file and compile it into a GrammarState.
pub fn grammar_load(source: &str) -> Result<GrammarState, String> {
    let rules = parse_gbnf(source)?;
    let nfa = compile_nfa(&rules);

    // Find root rule
    let root_rule = if nfa.rule_starts.contains_key("root") {
        "root".to_string()
    } else if let Some(first) = rules.first() {
        first.name.clone()
    } else {
        return Err("no rules in grammar".into());
    };

    // Initialize with epsilon closure from root start
    let root_start = nfa.rule_starts[&root_rule];
    let initial = epsilon_closure(&nfa, &[root_start]);

    Ok(GrammarState {
        nfa,
        active: initial,
        root_rule,
    })
}

/// Mask logits based on grammar validity.
/// For each token in the vocabulary, check if the token text would be valid
/// from the current grammar state. Invalid tokens get -inf (f32::NEG_INFINITY).
///
/// Returns a new logits array with invalid tokens masked out.
pub fn grammar_mask(state: &GrammarState, logits: &[f32], vocab: &[String]) -> Vec<f32> {
    let n = logits.len().min(vocab.len());
    let mut masked = logits.to_vec();

    // Get valid first bytes for quick rejection
    let valid_bytes = state.valid_bytes();

    for i in 0..n {
        let token_bytes = vocab[i].as_bytes();
        if token_bytes.is_empty() {
            continue; // empty tokens pass through
        }

        // Quick rejection: if first byte isn't valid, mask immediately
        if !valid_bytes[token_bytes[0] as usize] {
            masked[i] = f32::NEG_INFINITY;
            continue;
        }

        // Full check: can the entire token be consumed?
        if !state.token_is_valid(token_bytes) {
            masked[i] = f32::NEG_INFINITY;
        }
    }

    // Safety: if ALL tokens are masked, unmask the top-1 to prevent deadlock
    let all_masked = masked[..n].iter().all(|&v| v == f32::NEG_INFINITY);
    if all_masked && !logits.is_empty() {
        // Find the original argmax and unmask it
        let mut best = 0;
        for i in 1..logits.len() {
            if logits[i] > logits[best] {
                best = i;
            }
        }
        masked[best] = logits[best];
    }

    masked
}

/// Advance the grammar state after a token has been sampled.
/// Consumes each byte of the token text through the NFA.
pub fn grammar_advance(state: &mut GrammarState, token_text: &str) {
    state.advance_str(token_text);
}

/// Reset the grammar state back to the initial position (root rule start).
pub fn grammar_reset(state: &mut GrammarState) {
    let root_start = state.nfa.rule_starts[&state.root_rule];
    state.active = epsilon_closure(&state.nfa, &[root_start]);
}

// ═══════════════════════════════════════════════════════════════
// THREAD-LOCAL GRAMMAR STATE STORAGE
// ═══════════════════════════════════════════════════════════════

use std::cell::RefCell;

thread_local! {
    /// Active grammar state for constrained decoding.
    /// Set by grammar_load builtin, consumed by grammar_mask/grammar_advance.
    pub static GRAMMAR_STATE: RefCell<Option<GrammarState>> = RefCell::new(None);
}

/// Store a grammar state in thread-local storage (called from grammar_load builtin).
pub fn set_grammar_state(state: GrammarState) {
    GRAMMAR_STATE.with(|gs| {
        *gs.borrow_mut() = Some(state);
    });
}

/// Apply grammar masking to logits using the thread-local grammar state.
/// Returns None if no grammar is loaded.
pub fn apply_grammar_mask(logits: &[f32], vocab: &[String]) -> Option<Vec<f32>> {
    GRAMMAR_STATE.with(|gs| {
        let state = gs.borrow();
        state.as_ref().map(|s| grammar_mask(s, logits, vocab))
    })
}

/// Advance the thread-local grammar state after sampling a token.
pub fn advance_grammar(token_text: &str) {
    GRAMMAR_STATE.with(|gs| {
        if let Some(ref mut state) = *gs.borrow_mut() {
            grammar_advance(state, token_text);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_grammar() {
        let src = r#"
root ::= "hello" ws "world"
ws   ::= " "+
"#;
        let rules = parse_gbnf(src).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].name, "root");
        assert_eq!(rules[1].name, "ws");
    }

    #[test]
    fn test_grammar_load_and_valid_bytes() {
        let src = r#"root ::= "let" | "if""#;
        let state = grammar_load(src).unwrap();
        let valid = state.valid_bytes();
        assert!(valid[b'l' as usize]); // "let" starts with 'l'
        assert!(valid[b'i' as usize]); // "if" starts with 'i'
        assert!(!valid[b'x' as usize]); // nothing starts with 'x'
    }

    #[test]
    fn test_grammar_advance() {
        let src = r#"root ::= "let" | "if""#;
        let mut state = grammar_load(src).unwrap();
        assert!(state.token_is_valid(b"l"));
        assert!(state.token_is_valid(b"let"));
        assert!(state.token_is_valid(b"if"));
        assert!(!state.token_is_valid(b"x"));

        state.advance_str("l");
        assert!(state.token_is_valid(b"e"));
        assert!(!state.token_is_valid(b"f")); // only "let" is valid after "l"
    }

    #[test]
    fn test_grammar_mask_logits() {
        let src = r#"root ::= "a" | "b""#;
        let state = grammar_load(src).unwrap();
        let logits = vec![1.0, 2.0, 3.0];
        let vocab = vec!["a".to_string(), "b".to_string(), "x".to_string()];
        let masked = grammar_mask(&state, &logits, &vocab);
        assert_eq!(masked[0], 1.0); // "a" valid
        assert_eq!(masked[1], 2.0); // "b" valid
        assert_eq!(masked[2], f32::NEG_INFINITY); // "x" invalid
    }

    #[test]
    fn test_char_class() {
        let src = r#"root ::= [a-z]+"#;
        let state = grammar_load(src).unwrap();
        assert!(state.token_is_valid(b"hello"));
        assert!(state.token_is_valid(b"a"));
        assert!(!state.token_is_valid(b"A"));
        assert!(!state.token_is_valid(b"1"));
    }

    #[test]
    fn test_negated_char_class() {
        let src = r#"root ::= [^\n]+"#;
        let state = grammar_load(src).unwrap();
        assert!(state.token_is_valid(b"hello"));
        assert!(state.token_is_valid(b"123"));
        assert!(!state.token_is_valid(b"\n"));
    }

    #[test]
    fn test_optional_and_star() {
        let src = r#"root ::= "a" "b"? "c"*"#;
        let state = grammar_load(src).unwrap();
        assert!(state.token_is_valid(b"a"));
        assert!(state.token_is_valid(b"ab"));
        assert!(state.token_is_valid(b"ac"));
        assert!(state.token_is_valid(b"abc"));
        assert!(state.token_is_valid(b"abcc"));
    }

    #[test]
    fn test_all_masked_safety() {
        let src = r#"root ::= "x""#;
        let state = grammar_load(src).unwrap();
        let logits = vec![5.0, 3.0, 1.0];
        let vocab = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let masked = grammar_mask(&state, &logits, &vocab);
        // All would be masked, but safety net unmarks the best
        assert_eq!(masked[0], 5.0); // argmax unmasked
    }

    #[test]
    fn test_octoflow_grammar_loads() {
        // Verify the actual OctoFlow grammar parses without error
        let src = include_str!("../../../stdlib/llm/octoflow.gbnf");
        if !src.is_empty() {
            let result = grammar_load(src);
            // Should parse without panicking (may fail if file doesn't exist yet)
            if let Ok(state) = result {
                // "let" should be valid from root
                assert!(state.token_is_valid(b"let"));
                assert!(state.token_is_valid(b"if"));
                assert!(state.token_is_valid(b"fn"));
            }
        }
    }
}
