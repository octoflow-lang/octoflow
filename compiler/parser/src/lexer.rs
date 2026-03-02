//! Lexer for the OctoFlow v0.0.1 language.
//!
//! Produces a stream of [`SpannedToken`]s from source text.

/// A token with source location.
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub line: usize,
    pub col: usize,
}

/// Token types for the OctoFlow language.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Stream,
    Tap,
    RandomStream,
    Cache,
    Emit,
    Let,
    Print,
    Fn,
    Use,
    Struct,
    Mut,
    While,
    For,
    In,
    End,
    Break,
    Continue,
    If,
    Then,
    Else,
    Elif,
    Return,
    True,
    False,
    None,

    // Literals
    Ident(String),
    StringLit(String),
    NumberLit(f64),
    IntLit(i64),

    // Symbols
    Equals,
    Pipe, // |>
    LParen,
    RParen,
    Comma,
    Colon,
    Dot,
    LBracket,
    RBracket,

    // Arithmetic operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent, // %

    // Comparison operators
    Greater,      // >
    Less,         // <
    GreaterEqual, // >=
    LessEqual,    // <=
    EqualEqual,   // ==
    NotEqual,     // !=

    // Boolean operators
    And, // &&
    Or,  // ||

    // Bitwise operators
    Shl,    // <<
    Shr,    // >>
    BitAnd, // &
    BitOr,  // |
    BitXor, // ^

    // FFI / extern block tokens (Phase 44)
    Extern,  // extern keyword
    LBrace,  // {
    RBrace,  // }
    Arrow,   // ->

    // Meta
    Newline,
    Eof,
}

/// Tokenize source text into a vector of spanned tokens.
pub fn tokenize(source: &str) -> Result<Vec<SpannedToken>, LexError> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut pos = 0;
    let mut line = 1;
    let mut col = 1;

    while pos < chars.len() {
        let ch = chars[pos];

        // Skip line comments
        if ch == '/' && pos + 1 < chars.len() && chars[pos + 1] == '/' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }

        // Newline
        if ch == '\n' {
            // Only push Newline if the last token wasn't already a Newline/start
            if let Some(last) = tokens.last() {
                let last: &SpannedToken = last;
                if last.token != Token::Newline {
                    tokens.push(SpannedToken { token: Token::Newline, line, col });
                }
            }
            pos += 1;
            line += 1;
            col = 1;
            continue;
        }

        // Skip whitespace (not newlines)
        if ch == ' ' || ch == '\t' || ch == '\r' {
            pos += 1;
            col += 1;
            continue;
        }

        // Pipe operator |>
        if ch == '|' && pos + 1 < chars.len() && chars[pos + 1] == '>' {
            tokens.push(SpannedToken { token: Token::Pipe, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Boolean OR ||
        if ch == '|' && pos + 1 < chars.len() && chars[pos + 1] == '|' {
            tokens.push(SpannedToken { token: Token::Or, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Single | (bitwise OR) — after |> and || are handled
        if ch == '|' {
            tokens.push(SpannedToken { token: Token::BitOr, line, col });
            pos += 1;
            col += 1;
            continue;
        }

        // Boolean AND &&
        if ch == '&' && pos + 1 < chars.len() && chars[pos + 1] == '&' {
            tokens.push(SpannedToken { token: Token::And, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Single & (bitwise AND) — after && is handled
        if ch == '&' {
            tokens.push(SpannedToken { token: Token::BitAnd, line, col });
            pos += 1;
            col += 1;
            continue;
        }

        // Bitwise XOR ^
        if ch == '^' {
            tokens.push(SpannedToken { token: Token::BitXor, line, col });
            pos += 1;
            col += 1;
            continue;
        }

        // Shift operators << >> (must check before single < > and <=, >=)
        if ch == '<' && pos + 1 < chars.len() && chars[pos + 1] == '<' {
            tokens.push(SpannedToken { token: Token::Shl, line, col });
            pos += 2;
            col += 2;
            continue;
        }
        if ch == '>' && pos + 1 < chars.len() && chars[pos + 1] == '>' {
            tokens.push(SpannedToken { token: Token::Shr, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Two-character comparison operators (must check before single-char)
        if ch == '=' && pos + 1 < chars.len() && chars[pos + 1] == '=' {
            tokens.push(SpannedToken { token: Token::EqualEqual, line, col });
            pos += 2;
            col += 2;
            continue;
        }
        if ch == '!' && pos + 1 < chars.len() && chars[pos + 1] == '=' {
            tokens.push(SpannedToken { token: Token::NotEqual, line, col });
            pos += 2;
            col += 2;
            continue;
        }
        if ch == '>' && pos + 1 < chars.len() && chars[pos + 1] == '=' {
            tokens.push(SpannedToken { token: Token::GreaterEqual, line, col });
            pos += 2;
            col += 2;
            continue;
        }
        if ch == '<' && pos + 1 < chars.len() && chars[pos + 1] == '=' {
            tokens.push(SpannedToken { token: Token::LessEqual, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Single-character symbols
        match ch {
            '=' => {
                tokens.push(SpannedToken { token: Token::Equals, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '>' => {
                tokens.push(SpannedToken { token: Token::Greater, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '<' => {
                tokens.push(SpannedToken { token: Token::Less, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '(' => {
                tokens.push(SpannedToken { token: Token::LParen, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            ')' => {
                tokens.push(SpannedToken { token: Token::RParen, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            ',' => {
                tokens.push(SpannedToken { token: Token::Comma, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            ':' => {
                tokens.push(SpannedToken { token: Token::Colon, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '.' => {
                tokens.push(SpannedToken { token: Token::Dot, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '[' => {
                tokens.push(SpannedToken { token: Token::LBracket, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            ']' => {
                tokens.push(SpannedToken { token: Token::RBracket, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '{' => {
                tokens.push(SpannedToken { token: Token::LBrace, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '}' => {
                tokens.push(SpannedToken { token: Token::RBrace, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '+' => {
                tokens.push(SpannedToken { token: Token::Plus, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '*' => {
                tokens.push(SpannedToken { token: Token::Star, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            '%' => {
                tokens.push(SpannedToken { token: Token::Percent, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            _ => {}
        }

        // String literal
        if ch == '"' {
            let start_col = col;
            pos += 1;
            col += 1;
            let mut s = String::new();
            while pos < chars.len() && chars[pos] != '"' {
                if chars[pos] == '\n' {
                    return Err(LexError {
                        message: "unterminated string literal".to_string(),
                        line,
                        col: start_col,
                    });
                }
                if chars[pos] == '\\' && pos + 1 < chars.len() {
                    let esc = chars[pos + 1];
                    match esc {
                        'n'  => { s.push('\n'); pos += 2; col += 2; continue; }
                        't'  => { s.push('\t'); pos += 2; col += 2; continue; }
                        'r'  => { s.push('\r'); pos += 2; col += 2; continue; }
                        '\\' => { s.push('\\'); pos += 2; col += 2; continue; }
                        '"'  => { s.push('"');  pos += 2; col += 2; continue; }
                        '0'  => { s.push('\0'); pos += 2; col += 2; continue; }
                        _    => { /* unknown escape — pass through for regex strings like \d */ }
                    }
                }
                s.push(chars[pos]);
                pos += 1;
                col += 1;
            }
            if pos >= chars.len() {
                return Err(LexError {
                    message: "unterminated string literal".to_string(),
                    line,
                    col: start_col,
                });
            }
            pos += 1; // skip closing "
            col += 1;
            tokens.push(SpannedToken { token: Token::StringLit(s), line, col: start_col });
            continue;
        }

        // Arrow '->' (return type in extern blocks) — must check before '-'
        if ch == '-' && pos + 1 < chars.len() && chars[pos + 1] == '>' {
            tokens.push(SpannedToken { token: Token::Arrow, line, col });
            pos += 2;
            col += 2;
            continue;
        }

        // Minus: binary operator or negative number
        if ch == '-' {
            // `-` is a binary Minus operator when the previous token is
            // `)`, Ident, NumberLit, or IntLit. Otherwise it's part of a negative number.
            let is_binary = tokens.last().map_or(false, |t| {
                matches!(t.token, Token::RParen | Token::Ident(_) | Token::NumberLit(_) | Token::IntLit(_))
            });
            if is_binary {
                tokens.push(SpannedToken { token: Token::Minus, line, col });
                pos += 1;
                col += 1;
                continue;
            }
            // Fall through to number literal (negative)
            if pos + 1 < chars.len() && (chars[pos + 1].is_ascii_digit() || chars[pos + 1] == '.') {
                let start_col = col;
                let start = pos;
                pos += 1; // consume '-'
                col += 1;
                let mut has_dot = false;
                while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '.') {
                    if chars[pos] == '.' { has_dot = true; }
                    pos += 1;
                    col += 1;
                }
                let num_str: String = chars[start..pos].iter().collect();
                if has_dot {
                    let value: f64 = num_str.parse().map_err(|_| LexError {
                        message: format!("invalid number: {}", num_str),
                        line,
                        col: start_col,
                    })?;
                    tokens.push(SpannedToken { token: Token::NumberLit(value), line, col: start_col });
                } else {
                    let value: i64 = num_str.parse().map_err(|_| LexError {
                        message: format!("invalid number: {}", num_str),
                        line,
                        col: start_col,
                    })?;
                    tokens.push(SpannedToken { token: Token::IntLit(value), line, col: start_col });
                }
                continue;
            }
            // Standalone `-` not followed by digit — treat as Minus operator
            tokens.push(SpannedToken { token: Token::Minus, line, col });
            pos += 1;
            col += 1;
            continue;
        }

        // `/` — Slash operator (comments handled earlier)
        if ch == '/' {
            tokens.push(SpannedToken { token: Token::Slash, line, col });
            pos += 1;
            col += 1;
            continue;
        }

        // Number literal (decimal or hex 0x...)
        if ch.is_ascii_digit() {
            let start_col = col;
            // Hex literal: 0x or 0X
            if ch == '0' && pos + 1 < chars.len() && (chars[pos + 1] == 'x' || chars[pos + 1] == 'X') {
                pos += 2; // skip "0x"
                col += 2;
                let hex_start = pos;
                while pos < chars.len() && chars[pos].is_ascii_hexdigit() {
                    pos += 1;
                    col += 1;
                }
                if pos == hex_start {
                    return Err(LexError {
                        message: "expected hex digits after 0x".to_string(),
                        line,
                        col: start_col,
                    });
                }
                let hex_str: String = chars[hex_start..pos].iter().collect();
                let value = u64::from_str_radix(&hex_str, 16).map_err(|_| LexError {
                    message: format!("invalid hex literal: 0x{}", hex_str),
                    line,
                    col: start_col,
                })?;
                // Hex literals are always integers
                tokens.push(SpannedToken { token: Token::IntLit(value as i64), line, col: start_col });
                continue;
            }
            let start = pos;
            let mut has_dot = false;
            while pos < chars.len() && (chars[pos].is_ascii_digit() || chars[pos] == '.') {
                if chars[pos] == '.' { has_dot = true; }
                pos += 1;
                col += 1;
            }
            let num_str: String = chars[start..pos].iter().collect();
            if has_dot {
                let value: f64 = num_str.parse().map_err(|_| LexError {
                    message: format!("invalid number: {}", num_str),
                    line,
                    col: start_col,
                })?;
                tokens.push(SpannedToken { token: Token::NumberLit(value), line, col: start_col });
            } else {
                let value: i64 = num_str.parse().map_err(|_| LexError {
                    message: format!("invalid number: {}", num_str),
                    line,
                    col: start_col,
                })?;
                tokens.push(SpannedToken { token: Token::IntLit(value), line, col: start_col });
            }
            continue;
        }

        // Identifier or keyword
        if ch.is_ascii_alphabetic() || ch == '_' {
            let start_col = col;
            let start = pos;
            while pos < chars.len() && (chars[pos].is_ascii_alphanumeric() || chars[pos] == '_') {
                pos += 1;
                col += 1;
            }
            let word: String = chars[start..pos].iter().collect();
            let token = match word.as_str() {
                "stream" => Token::Stream,
                "tap" => Token::Tap,
                "random_stream" => Token::RandomStream,
                "cache" => Token::Cache,
                "emit" => Token::Emit,
                "let" => Token::Let,
                "print" => Token::Print,
                "fn" => Token::Fn,
                "use" => Token::Use,
                "struct" => Token::Struct,
                "mut" => Token::Mut,
                "while" => Token::While,
                "for" => Token::For,
                "in" => Token::In,
                "end" => Token::End,
                "break" => Token::Break,
                "continue" => Token::Continue,
                "return" => Token::Return,
                "if" => Token::If,
                "then" => Token::Then,
                "else" => Token::Else,
                "elif" => Token::Elif,
                "true" => Token::True,
                "false" => Token::False,
                "none" => Token::None,
                "extern" => Token::Extern,
                _ => Token::Ident(word),
            };
            tokens.push(SpannedToken { token, line, col: start_col });
            continue;
        }

        return Err(LexError {
            message: format!("unexpected character: '{}'", ch),
            line,
            col,
        });
    }

    // Ensure trailing Newline is present for the last statement
    if let Some(last) = tokens.last() {
        if last.token != Token::Newline {
            tokens.push(SpannedToken { token: Token::Newline, line, col });
        }
    }

    tokens.push(SpannedToken { token: Token::Eof, line, col });
    Ok(tokens)
}

/// Lexer error with source location.
#[derive(Debug, Clone)]
pub struct LexError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, col {}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for LexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_stream_decl() {
        let tokens = tokenize("stream x = tap(\"a.csv\")\n").unwrap();
        assert!(matches!(tokens[0].token, Token::Stream));
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "x"));
        assert!(matches!(tokens[2].token, Token::Equals));
        assert!(matches!(tokens[3].token, Token::Tap));
        assert!(matches!(tokens[4].token, Token::LParen));
        assert!(matches!(&tokens[5].token, Token::StringLit(s) if s == "a.csv"));
        assert!(matches!(tokens[6].token, Token::RParen));
        assert!(matches!(tokens[7].token, Token::Newline));
    }

    #[test]
    fn test_lex_pipe() {
        let tokens = tokenize("x |> multiply(2.0)\n").unwrap();
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "x"));
        assert!(matches!(tokens[1].token, Token::Pipe));
        assert!(matches!(&tokens[2].token, Token::Ident(s) if s == "multiply"));
        assert!(matches!(tokens[3].token, Token::LParen));
        assert!(matches!(&tokens[4].token, Token::NumberLit(n) if (*n - 2.0).abs() < 1e-10));
        assert!(matches!(tokens[5].token, Token::RParen));
    }

    #[test]
    fn test_lex_negative_number() {
        let tokens = tokenize("add(-10.0)\n").unwrap();
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "add"));
        assert!(matches!(tokens[1].token, Token::LParen));
        assert!(matches!(&tokens[2].token, Token::NumberLit(n) if (*n - (-10.0)).abs() < 1e-10));
        assert!(matches!(tokens[3].token, Token::RParen));
    }

    #[test]
    fn test_lex_comment() {
        let tokens = tokenize("// this is a comment\nstream x = tap(\"a.csv\")\n").unwrap();
        // First real token should be Stream (comment skipped)
        assert!(matches!(tokens[0].token, Token::Stream));
    }

    #[test]
    fn test_lex_emit() {
        let tokens = tokenize("emit(result, \"out.csv\")\n").unwrap();
        assert!(matches!(tokens[0].token, Token::Emit));
        assert!(matches!(tokens[1].token, Token::LParen));
        assert!(matches!(&tokens[2].token, Token::Ident(s) if s == "result"));
        assert!(matches!(tokens[3].token, Token::Comma));
        assert!(matches!(&tokens[4].token, Token::StringLit(s) if s == "out.csv"));
        assert!(matches!(tokens[5].token, Token::RParen));
    }

    #[test]
    fn test_lex_unterminated_string() {
        let result = tokenize("tap(\"unclosed\n");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("unterminated"));
    }

    #[test]
    fn test_lex_unknown_char() {
        let result = tokenize("stream x = @bad\n");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("unexpected character"));
    }

    #[test]
    fn test_lex_let_keyword() {
        let tokens = tokenize("let x = 5.0\n").unwrap();
        assert!(matches!(tokens[0].token, Token::Let));
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "x"));
        assert!(matches!(tokens[2].token, Token::Equals));
        assert!(matches!(&tokens[3].token, Token::NumberLit(n) if (*n - 5.0).abs() < 1e-10));
    }

    #[test]
    fn test_lex_print_keyword() {
        let tokens = tokenize("print(\"hello\")\n").unwrap();
        assert!(matches!(tokens[0].token, Token::Print));
        assert!(matches!(tokens[1].token, Token::LParen));
        assert!(matches!(&tokens[2].token, Token::StringLit(s) if s == "hello"));
        assert!(matches!(tokens[3].token, Token::RParen));
    }

    #[test]
    fn test_lex_arithmetic_ops() {
        let tokens = tokenize("mx - mn\n").unwrap();
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "mx"));
        assert!(matches!(tokens[1].token, Token::Minus));
        assert!(matches!(&tokens[2].token, Token::Ident(s) if s == "mn"));
    }

    #[test]
    fn test_lex_arithmetic_all_ops() {
        let tokens = tokenize("a + b * c / d\n").unwrap();
        assert!(matches!(tokens[1].token, Token::Plus));
        assert!(matches!(tokens[3].token, Token::Star));
        assert!(matches!(tokens[5].token, Token::Slash));
    }

    #[test]
    fn test_lex_minus_after_paren_is_binary() {
        // After `)`, `-` should be a Minus operator, not negative literal
        let tokens = tokenize("min(data) - 1.0\n").unwrap();
        // min ( data ) - 1.0
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "min"));
        assert!(matches!(tokens[3].token, Token::RParen));
        assert!(matches!(tokens[4].token, Token::Minus));
        assert!(matches!(&tokens[5].token, Token::NumberLit(n) if (*n - 1.0).abs() < 1e-10));
    }

    #[test]
    fn test_lex_comparison_ops() {
        let tokens = tokenize("a > b\n").unwrap();
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "a"));
        assert!(matches!(tokens[1].token, Token::Greater));

        let tokens = tokenize("a >= b\n").unwrap();
        assert!(matches!(tokens[1].token, Token::GreaterEqual));

        let tokens = tokenize("a < b\n").unwrap();
        assert!(matches!(tokens[1].token, Token::Less));

        let tokens = tokenize("a <= b\n").unwrap();
        assert!(matches!(tokens[1].token, Token::LessEqual));

        let tokens = tokenize("a == b\n").unwrap();
        assert!(matches!(tokens[1].token, Token::EqualEqual));

        let tokens = tokenize("a != b\n").unwrap();
        assert!(matches!(tokens[1].token, Token::NotEqual));
    }

    #[test]
    fn test_lex_boolean_ops() {
        let tokens = tokenize("a && b || c\n").unwrap();
        assert!(matches!(&tokens[0].token, Token::Ident(s) if s == "a"));
        assert!(matches!(tokens[1].token, Token::And));
        assert!(matches!(&tokens[2].token, Token::Ident(s) if s == "b"));
        assert!(matches!(tokens[3].token, Token::Or));
        assert!(matches!(&tokens[4].token, Token::Ident(s) if s == "c"));
    }

    #[test]
    fn test_lex_if_then_else_keywords() {
        let tokens = tokenize("if x then 1.0 else 0.0\n").unwrap();
        assert!(matches!(tokens[0].token, Token::If));
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "x"));
        assert!(matches!(tokens[2].token, Token::Then));
        assert!(matches!(&tokens[3].token, Token::NumberLit(n) if (*n - 1.0).abs() < 1e-10));
        assert!(matches!(tokens[4].token, Token::Else));
        assert!(matches!(&tokens[5].token, Token::NumberLit(n) if (*n - 0.0).abs() < 1e-10));
    }

    #[test]
    fn test_lex_true_false() {
        let tokens = tokenize("true false\n").unwrap();
        assert!(matches!(tokens[0].token, Token::True));
        assert!(matches!(tokens[1].token, Token::False));
    }

    #[test]
    fn test_lex_while_end() {
        let tokens = tokenize("while x > 0.0\nend\n").unwrap();
        assert!(matches!(tokens[0].token, Token::While));
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "x"));
        assert!(matches!(tokens[2].token, Token::Greater));
        // After newline and body, we get End
        let end_pos = tokens.iter().position(|t| matches!(t.token, Token::End)).unwrap();
        assert!(end_pos > 0);
    }

    #[test]
    fn test_lex_for_in() {
        let tokens = tokenize("for i in range(0, 10)\nend\n").unwrap();
        assert!(matches!(tokens[0].token, Token::For));
        assert!(matches!(&tokens[1].token, Token::Ident(s) if s == "i"));
        assert!(matches!(tokens[2].token, Token::In));
        assert!(matches!(&tokens[3].token, Token::Ident(s) if s == "range"));
        let end_pos = tokens.iter().position(|t| matches!(t.token, Token::End)).unwrap();
        assert!(end_pos > 0);
    }

    #[test]
    fn test_string_escape_newline() {
        let tokens = tokenize("let x = \"hello\\nworld\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "hello\nworld"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_escape_tab() {
        let tokens = tokenize("let x = \"col1\\tcol2\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "col1\tcol2"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_escape_backslash() {
        let tokens = tokenize("let x = \"path\\\\file\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "path\\file"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_escape_quote() {
        let tokens = tokenize("let x = \"say \\\"hi\\\"\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "say \"hi\""),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_escape_carriage_return() {
        let tokens = tokenize("let x = \"line\\r\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "line\r"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_escape_null() {
        let tokens = tokenize("let x = \"end\\0\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "end\0"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_unknown_escape_passthrough() {
        // Unknown escapes (like \d for regex) pass through unchanged
        let tokens = tokenize("let x = \"\\d+\"\n").unwrap();
        match &tokens[3].token {
            Token::StringLit(s) => assert_eq!(s, "\\d+"),
            other => panic!("expected StringLit, got {:?}", other),
        }
    }
}
