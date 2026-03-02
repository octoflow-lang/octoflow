//! OctoFlow parser — hand-written recursive descent, no dependencies.
//!
//! Entry point: [`parse`] takes source text and returns a [`Program`] AST.

pub mod ast;
pub mod lexer;

use ast::{Arg, BinOp, CompareOp, Expr, ExternFn, ExternParam, PrintSegment, Program, ScalarExpr, Span, StageCall, Statement};
use lexer::{SpannedToken, Token, tokenize};

/// Parse error with source location.
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}, col {}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for ParseError {}

/// Parse OctoFlow source text into a Program AST.
pub fn parse(source: &str) -> Result<Program, ParseError> {
    let tokens = tokenize(source).map_err(|e| ParseError {
        message: e.message,
        line: e.line,
        col: e.col,
    })?;
    let mut parser = Parser::new(tokens);
    parser.parse_program()
}

/// Parse a single scalar expression (for REPL bare-expression evaluation).
pub fn parse_expr(source: &str) -> Result<ScalarExpr, ParseError> {
    let tokens = tokenize(source).map_err(|e| ParseError {
        message: e.message,
        line: e.line,
        col: e.col,
    })?;
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_scalar_expr()?;
    // Skip trailing newlines
    while matches!(parser.peek(), Token::Newline) {
        parser.advance();
    }
    if !matches!(parser.peek(), Token::Eof) {
        return Err(parser.error("unexpected token after expression"));
    }
    Ok(expr)
}

struct Parser {
    tokens: Vec<SpannedToken>,
    pos: usize,
}

impl Parser {
    fn new(tokens: Vec<SpannedToken>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        if self.pos < self.tokens.len() {
            &self.tokens[self.pos].token
        } else {
            &Token::Eof
        }
    }

    fn current_span(&self) -> (usize, usize) {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos];
            (t.line, t.col)
        } else {
            (0, 0)
        }
    }

    fn advance(&mut self) -> &SpannedToken {
        if self.pos < self.tokens.len() {
            let t = &self.tokens[self.pos];
            self.pos += 1;
            t
        } else {
            // Past end of tokens — return last token (should be EOF)
            &self.tokens[self.tokens.len() - 1]
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<&SpannedToken, ParseError> {
        let (line, col) = self.current_span();
        let actual = self.peek().clone();
        if std::mem::discriminant(&actual) == std::mem::discriminant(expected) {
            Ok(self.advance())
        } else {
            Err(ParseError {
                message: format!("expected {:?}, got {:?}", expected, actual),
                line,
                col,
            })
        }
    }

    fn error(&self, msg: &str) -> ParseError {
        let (line, col) = self.current_span();
        ParseError {
            message: msg.to_string(),
            line,
            col,
        }
    }

    /// Consume a trailing newline or EOF after a statement.
    fn expect_newline(&mut self) -> Result<(), ParseError> {
        match self.peek() {
            Token::Newline | Token::Eof => {
                if matches!(self.peek(), Token::Newline) {
                    self.advance();
                }
                Ok(())
            }
            _ => Err(self.error("expected newline after statement")),
        }
    }

    // ── Grammar rules ───────────────────────────────────────────────

    /// program → statement* EOF
    fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut statements = Vec::new();

        // Skip leading newlines
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }

        while !matches!(self.peek(), Token::Eof) {
            let (line, col) = self.current_span();
            let stmt = self.parse_statement()?;
            statements.push((stmt, Span { line, col }));

            // Skip newlines between statements
            while matches!(self.peek(), Token::Newline) {
                self.advance();
            }
        }

        Ok(Program { statements })
    }

    /// statement → stream_decl | let_decl | emit_stmt | print_stmt | fn_decl | use_decl | assign | extern_block
    fn parse_statement(&mut self) -> Result<Statement, ParseError> {
        match self.peek() {
            Token::Stream => self.parse_stream_decl(),
            Token::Let => self.parse_let_decl(),
            Token::Emit => self.parse_emit(),
            Token::Print => self.parse_print(),
            Token::Fn => self.parse_fn_decl(),
            Token::Use => self.parse_use_decl(),
            Token::Struct => self.parse_struct_def(),
            Token::While => self.parse_while(),
            Token::For => self.parse_for(),
            Token::If => self.parse_if_block(),
            Token::Extern => self.parse_extern_block(),
            Token::Break => {
                self.advance(); // consume "break"
                self.expect_newline()?;
                Ok(Statement::Break)
            }
            Token::Continue => {
                self.advance(); // consume "continue"
                self.expect_newline()?;
                Ok(Statement::Continue)
            }
            Token::Return => {
                self.advance(); // consume "return"
                let value = self.parse_scalar_expr()?;
                self.expect_newline()?;
                Ok(Statement::Return { value })
            }
            Token::Ident(ref s) if s == "push" => {
                self.parse_push()
            }
            Token::Ident(ref s) if s == "map_set" => {
                self.parse_map_set()
            }
            Token::Ident(ref s) if s == "write_file" => {
                self.parse_file_write("write_file")
            }
            Token::Ident(ref s) if s == "append_file" => {
                self.parse_file_write("append_file")
            }
            Token::Ident(ref s) if s == "save_data" => {
                self.parse_save_data()
            }
            Token::Ident(ref s) if s == "write_csv" => {
                self.parse_write_csv()
            }
            Token::Ident(ref s) if s == "write_bytes" => {
                self.parse_write_bytes()
            }
            Token::Ident(_) => {
                // Peek ahead: `=` → assignment, `[` → array element assign, `(` → bare fn call
                if self.pos + 1 < self.tokens.len() {
                    match &self.tokens[self.pos + 1].token {
                        Token::Equals => self.parse_assign(),
                        Token::LBracket => self.parse_array_assign(),
                        Token::LParen => {
                            // Bare function call statement: name(args...)
                            let expr = self.parse_scalar_expr()?;
                            self.expect_newline()?;
                            Ok(Statement::ExprStmt { expr })
                        }
                        _ => Err(self.error("expected assignment, array access, or function call")),
                    }
                } else {
                    Err(self.error("unexpected end of input"))
                }
            }
            _ => Err(self.error("expected statement")),
        }
    }

    /// assign → IDENT "=" scalar_expr NEWLINE
    fn parse_assign(&mut self) -> Result<Statement, ParseError> {
        let name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected variable name")),
        };
        self.expect(&Token::Equals)?;
        let value = self.parse_scalar_expr()?;
        self.expect_newline()?;
        Ok(Statement::Assign { name, value })
    }

    /// array_assign → IDENT "[" scalar_expr "]" "=" scalar_expr NEWLINE
    fn parse_array_assign(&mut self) -> Result<Statement, ParseError> {
        let array = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected array name")),
        };
        self.expect(&Token::LBracket)?;
        let index = self.parse_scalar_expr()?;
        self.expect(&Token::RBracket)?;
        self.expect(&Token::Equals)?;
        let value = self.parse_scalar_expr()?;
        self.expect_newline()?;
        Ok(Statement::ArrayAssign { array, index, value })
    }

    /// push_stmt → "push" "(" IDENT "," scalar_expr ")" NEWLINE
    fn parse_push(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "push"
        self.expect(&Token::LParen)?;
        let array = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected array name as first argument to push()")),
        };
        self.expect(&Token::Comma)?;
        let value = self.parse_scalar_expr()?;
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        Ok(Statement::ArrayPush { array, value })
    }

    /// map_set_stmt → "map_set" "(" IDENT "," scalar_expr "," scalar_expr ")" NEWLINE
    fn parse_map_set(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "map_set"
        self.expect(&Token::LParen)?;
        let map = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected map name as first argument to map_set()")),
        };
        self.expect(&Token::Comma)?;
        let key = self.parse_scalar_expr()?;
        self.expect(&Token::Comma)?;
        let value = self.parse_scalar_expr()?;
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        Ok(Statement::MapInsert { map, key, value })
    }

    /// file_write_stmt → ("write_file"|"append_file") "(" scalar_expr "," scalar_expr ")" NEWLINE
    fn parse_file_write(&mut self, kind: &str) -> Result<Statement, ParseError> {
        self.advance(); // consume "write_file" or "append_file"
        self.expect(&Token::LParen)?;
        let path = self.parse_scalar_expr()?;
        self.expect(&Token::Comma)?;
        let content = self.parse_scalar_expr()?;
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        if kind == "write_file" {
            Ok(Statement::WriteFile { path, content })
        } else {
            Ok(Statement::AppendFile { path, content })
        }
    }

    /// save_data_stmt → "save_data" "(" scalar_expr "," IDENT ")" NEWLINE
    fn parse_save_data(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "save_data"
        self.expect(&Token::LParen)?;
        let path = self.parse_scalar_expr()?;
        self.expect(&Token::Comma)?;
        let map_name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected map name as second argument to save_data()")),
        };
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        Ok(Statement::SaveData { path, map_name })
    }

    /// write_csv_stmt → "write_csv" "(" scalar_expr "," IDENT ")" NEWLINE
    fn parse_write_csv(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "write_csv"
        self.expect(&Token::LParen)?;
        let path = self.parse_scalar_expr()?;
        self.expect(&Token::Comma)?;
        let array_name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected array name as second argument to write_csv()")),
        };
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        Ok(Statement::WriteCsv { path, array_name })
    }

    /// write_bytes_stmt → "write_bytes" "(" scalar_expr "," IDENT ")" NEWLINE
    fn parse_write_bytes(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "write_bytes"
        self.expect(&Token::LParen)?;
        let path = self.parse_scalar_expr()?;
        self.expect(&Token::Comma)?;
        let array_name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected array name as second argument to write_bytes()")),
        };
        self.expect(&Token::RParen)?;
        self.expect_newline()?;
        Ok(Statement::WriteBytes { path, array_name })
    }

    /// stream_decl → "stream" IDENT "=" expr NEWLINE
    fn parse_stream_decl(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "stream"

        let name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected stream name")),
        };

        self.expect(&Token::Equals)?;

        let expr = self.parse_expr()?;
        self.expect_newline()?;

        Ok(Statement::StreamDecl { name, expr })
    }

    /// let_decl → "let" "mut"? IDENT "=" (array_lit | scalar_expr) NEWLINE
    fn parse_let_decl(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "let"

        // Check for `mut` keyword
        let mutable = matches!(self.peek(), Token::Mut);
        if mutable {
            self.advance(); // consume "mut"
        }

        let name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected variable name after 'let'")),
        };

        self.expect(&Token::Equals)?;

        // Check for map constructor: let m = map()
        if matches!(self.peek(), Token::Ident(ref s) if s == "map") {
            // Peek ahead for '(' ')' to confirm it's map()
            if self.pos + 2 < self.tokens.len()
                && matches!(self.tokens[self.pos + 1].token, Token::LParen)
                && matches!(self.tokens[self.pos + 2].token, Token::RParen)
            {
                self.advance(); // consume "map"
                self.advance(); // consume '('
                self.advance(); // consume ')'
                self.expect_newline()?;
                return Ok(Statement::MapDecl { name, mutable });
            }
        }

        // Check for array literal: let arr = [expr, expr, ...]
        if matches!(self.peek(), Token::LBracket) {
            self.advance(); // consume '['
            let mut elements = Vec::new();
            if !matches!(self.peek(), Token::RBracket) {
                elements.push(self.parse_scalar_expr()?);
                while matches!(self.peek(), Token::Comma) {
                    self.advance(); // consume ','
                    elements.push(self.parse_scalar_expr()?);
                }
            }
            self.expect(&Token::RBracket)?;
            self.expect_newline()?;
            return Ok(Statement::ArrayDecl { name, elements, mutable });
        }

        let value = self.parse_scalar_expr()?;
        self.expect_newline()?;

        Ok(Statement::LetDecl { name, value, mutable })
    }

    /// emit_stmt → "emit" "(" expr "," STRING ")" NEWLINE
    fn parse_emit(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "emit"
        self.expect(&Token::LParen)?;

        let expr = self.parse_expr()?;

        self.expect(&Token::Comma)?;

        let path = match self.peek().clone() {
            Token::StringLit(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected output file path string")),
        };

        self.expect(&Token::RParen)?;
        self.expect_newline()?;

        Ok(Statement::Emit { expr, path })
    }

    /// print_stmt → "print" "(" STRING ")" NEWLINE
    ///
    /// The string may contain `{name}` or `{name:.N}` interpolation segments.
    fn parse_print(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "print"
        self.expect(&Token::LParen)?;

        let segments = match self.peek().clone() {
            Token::StringLit(s) => {
                self.advance();
                parse_interpolation(&s)
                    .map_err(|e| self.error(&e))?
            }
            _ => {
                // Expression form: print(expr)
                let expr = self.parse_scalar_expr()?;
                vec![PrintSegment::Expr(expr)]
            }
        };

        self.expect(&Token::RParen)?;
        self.expect_newline()?;

        Ok(Statement::Print { segments })
    }

    /// fn_decl → "fn" IDENT "(" (IDENT ("," IDENT)*)? ")" ":" NEWLINE? stage_call ("|>" stage_call)* NEWLINE
    fn parse_fn_decl(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "fn"

        let name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected function name after 'fn'")),
        };

        self.expect(&Token::LParen)?;

        let mut params = Vec::new();
        if matches!(self.peek(), Token::Ident(_)) {
            if let Token::Ident(p) = self.peek().clone() {
                self.advance();
                params.push(p);
            }
            while matches!(self.peek(), Token::Comma) {
                self.advance();
                match self.peek().clone() {
                    Token::Ident(p) => {
                        self.advance();
                        params.push(p);
                    }
                    _ => return Err(self.error("expected parameter name")),
                }
            }
        }

        self.expect(&Token::RParen)?;

        // Disambiguate: ':' → pipeline fn, newline → scalar fn body until 'end'
        if matches!(self.peek(), Token::Colon) {
            self.advance(); // consume ':'

            // Skip optional newline after colon (body can be on next line)
            if matches!(self.peek(), Token::Newline) {
                self.advance();
            }

            // Parse pipeline body: stage_call ("|>" stage_call)*
            let mut body = Vec::new();
            body.push(self.parse_stage_call()?);
            while matches!(self.peek(), Token::Pipe) {
                self.advance();
                body.push(self.parse_stage_call()?);
            }

            self.expect_newline()?;

            Ok(Statement::FnDecl { name, params, body })
        } else {
            // Scalar function with imperative body
            self.expect_newline()?;
            let body = self.parse_block_body(&[Token::End])?;
            if matches!(self.peek(), Token::Eof) {
                return Err(self.error("expected 'end' to close 'fn' block"));
            }
            self.advance(); // consume "end"
            self.expect_newline()?;
            Ok(Statement::ScalarFnDecl { name, params, body })
        }
    }

    /// use_decl → "use" IDENT NEWLINE
    fn parse_use_decl(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "use"

        let module = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            Token::StringLit(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected module name after 'use'")),
        };

        self.expect_newline()?;

        Ok(Statement::UseDecl { module })
    }

    /// struct_def → "struct" IDENT "(" IDENT ("," IDENT)* ")" NEWLINE
    fn parse_struct_def(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "struct"

        let name = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected struct name after 'struct'")),
        };

        self.expect(&Token::LParen)?;

        let mut fields = Vec::new();
        // Parse first field
        match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                fields.push(s);
            }
            _ => return Err(self.error("expected field name")),
        }
        // Parse remaining fields
        while matches!(self.peek(), Token::Comma) {
            self.advance(); // consume ','
            match self.peek().clone() {
                Token::Ident(s) => {
                    self.advance();
                    fields.push(s);
                }
                _ => return Err(self.error("expected field name after ','")),
            }
        }

        self.expect(&Token::RParen)?;
        self.expect_newline()?;

        Ok(Statement::StructDef { name, fields })
    }

    /// while_loop → "while" scalar_expr NEWLINE statement* "end" NEWLINE
    fn parse_while(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "while"
        let condition = self.parse_scalar_expr()?;
        self.expect_newline()?;

        let mut body = Vec::new();
        // Skip blank lines before body
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
        while !matches!(self.peek(), Token::End | Token::Eof) {
            let (line, col) = self.current_span();
            let stmt = self.parse_statement()?;
            body.push((stmt, Span { line, col }));
            // Skip newlines between statements
            while matches!(self.peek(), Token::Newline) {
                self.advance();
            }
        }
        if matches!(self.peek(), Token::Eof) {
            return Err(self.error("expected 'end' to close 'while' block"));
        }
        self.advance(); // consume "end"
        self.expect_newline()?;

        Ok(Statement::WhileLoop { condition, body })
    }

    /// for_loop → "for" IDENT "in" ("range" "(" scalar_expr "," scalar_expr ")" | IDENT) NEWLINE statement* "end" NEWLINE
    fn parse_for(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "for"

        let var = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected variable name after 'for'")),
        };

        // Expect 'in' keyword
        if !matches!(self.peek(), Token::In) {
            return Err(self.error("expected 'in' after for-loop variable"));
        }
        self.advance(); // consume "in"

        // Disambiguate: 'range(...)' → ForLoop, other identifier → ForEachLoop
        match self.peek().clone() {
            Token::Ident(s) if s == "range" => {
                self.advance(); // consume "range"
                self.expect(&Token::LParen)?;
                let start = self.parse_scalar_expr()?;
                self.expect(&Token::Comma)?;
                let end = self.parse_scalar_expr()?;
                self.expect(&Token::RParen)?;
                self.expect_newline()?;

                let body = self.parse_for_body()?;
                Ok(Statement::ForLoop { var, start, end, body })
            }
            Token::Ident(iterable) => {
                self.advance(); // consume iterable name
                self.expect_newline()?;

                let body = self.parse_for_body()?;
                Ok(Statement::ForEachLoop { var, iterable, body })
            }
            _ => Err(self.error("expected 'range(...)' or array name after 'in'")),
        }
    }

    /// Helper: parse the body of a for/for-each loop (statements until 'end').
    fn parse_for_body(&mut self) -> Result<Vec<(Statement, Span)>, ParseError> {
        let mut body = Vec::new();
        // Skip blank lines before body
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
        while !matches!(self.peek(), Token::End | Token::Eof) {
            let (line, col) = self.current_span();
            let stmt = self.parse_statement()?;
            body.push((stmt, Span { line, col }));
            // Skip newlines between statements
            while matches!(self.peek(), Token::Newline) {
                self.advance();
            }
        }
        if matches!(self.peek(), Token::Eof) {
            return Err(self.error("expected 'end' to close 'for' block"));
        }
        self.advance(); // consume "end"
        self.expect_newline()?;
        Ok(body)
    }

    /// if_block → "if" scalar_expr NEWLINE statement* ("elif" scalar_expr NEWLINE statement*)* ("else" NEWLINE statement*)? "end" NEWLINE
    fn parse_if_block(&mut self) -> Result<Statement, ParseError> {
        self.advance(); // consume "if"
        let condition = self.parse_scalar_expr()?;
        self.expect_newline()?;

        // Parse if-body
        let body = self.parse_block_body(&[Token::Elif, Token::Else, Token::End])?;

        // Parse elif branches
        let mut elif_branches = Vec::new();
        while matches!(self.peek(), Token::Elif) {
            self.advance(); // consume "elif"
            let elif_cond = self.parse_scalar_expr()?;
            self.expect_newline()?;
            let elif_body = self.parse_block_body(&[Token::Elif, Token::Else, Token::End])?;
            elif_branches.push((elif_cond, elif_body));
        }

        // Parse optional else branch
        let else_body = if matches!(self.peek(), Token::Else) {
            self.advance(); // consume "else"
            self.expect_newline()?;
            self.parse_block_body(&[Token::End])?
        } else {
            Vec::new()
        };

        if matches!(self.peek(), Token::Eof) {
            return Err(self.error("expected 'end' to close 'if' block"));
        }
        self.advance(); // consume "end"
        self.expect_newline()?;

        Ok(Statement::IfBlock { condition, body, elif_branches, else_body })
    }

    /// Helper: parse statements until one of the terminator tokens is reached.
    fn parse_block_body(&mut self, terminators: &[Token]) -> Result<Vec<(Statement, Span)>, ParseError> {
        let mut body = Vec::new();
        // Skip blank lines
        while matches!(self.peek(), Token::Newline) {
            self.advance();
        }
        while !self.is_terminator(terminators) && !matches!(self.peek(), Token::Eof) {
            let (line, col) = self.current_span();
            let stmt = self.parse_statement()?;
            body.push((stmt, Span { line, col }));
            while matches!(self.peek(), Token::Newline) {
                self.advance();
            }
        }
        Ok(body)
    }

    /// Check if current token matches any of the terminator tokens.
    fn is_terminator(&self, terminators: &[Token]) -> bool {
        let current = self.peek();
        terminators.iter().any(|t| std::mem::discriminant(t) == std::mem::discriminant(current))
    }

    /// expr → primary ("|>" stage_call)*
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let primary = self.parse_primary()?;

        if !matches!(self.peek(), Token::Pipe) {
            return Ok(primary);
        }

        let mut stages = Vec::new();
        while matches!(self.peek(), Token::Pipe) {
            self.advance(); // consume |>
            stages.push(self.parse_stage_call()?);
        }

        Ok(Expr::Pipe {
            input: Box::new(primary),
            stages,
        })
    }

    /// stage_call → IDENT ("." IDENT)? "(" (arg ("," arg)*)? ")"
    fn parse_stage_call(&mut self) -> Result<StageCall, ParseError> {
        let mut operation = match self.peek().clone() {
            Token::Ident(s) => {
                self.advance();
                s
            }
            _ => return Err(self.error("expected operation name")),
        };

        // Handle dotted names: module.function
        if matches!(self.peek(), Token::Dot) {
            self.advance(); // consume '.'
            match self.peek().clone() {
                Token::Ident(s) => {
                    self.advance();
                    operation = format!("{}.{}", operation, s);
                }
                _ => return Err(self.error("expected function name after '.'")),
            }
        }

        self.expect(&Token::LParen)?;

        let mut args = Vec::new();
        if !matches!(self.peek(), Token::RParen) {
            args.push(self.parse_arg()?);
            while matches!(self.peek(), Token::Comma) {
                self.advance(); // consume comma
                args.push(self.parse_arg()?);
            }
        }

        self.expect(&Token::RParen)?;

        Ok(StageCall { operation, args })
    }

    /// arg → NUMBER | INT | IDENT
    fn parse_arg(&mut self) -> Result<Arg, ParseError> {
        match self.peek().clone() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(Arg::Literal(n))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(Arg::IntLiteral(n))
            }
            Token::Ident(s) => {
                self.advance();
                Ok(Arg::Ref(s))
            }
            _ => Err(self.error("expected number or identifier argument")),
        }
    }

    /// primary → tap_expr | IDENT
    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek().clone() {
            Token::Tap => {
                self.advance(); // consume "tap"
                self.expect(&Token::LParen)?;
                let path = match self.peek().clone() {
                    Token::StringLit(s) => {
                        self.advance();
                        s
                    }
                    _ => return Err(self.error("expected file path string in tap()")),
                };
                self.expect(&Token::RParen)?;
                Ok(Expr::Tap { path })
            }
            Token::RandomStream => {
                self.advance(); // consume "random_stream"
                self.expect(&Token::LParen)?;
                // Parse count as a scalar expression
                let count = self.parse_scalar_expr()?;
                // Optional lo, hi bounds (default: -1.0..1.0)
                let (lo, hi) = if matches!(self.peek(), Token::Comma) {
                    self.advance(); // consume ','
                    let lo_tok = self.peek().clone();
                    let lo = match lo_tok {
                        Token::NumberLit(n) => { self.advance(); n as f32 }
                        Token::IntLit(n) => { self.advance(); n as f32 }
                        _ => return Err(self.error("random_stream: expected number for lo bound")),
                    };
                    self.expect(&Token::Comma)?;
                    let hi_tok = self.peek().clone();
                    let hi = match hi_tok {
                        Token::NumberLit(n) => { self.advance(); n as f32 }
                        Token::IntLit(n) => { self.advance(); n as f32 }
                        _ => return Err(self.error("random_stream: expected number for hi bound")),
                    };
                    (lo, hi)
                } else {
                    (-1.0f32, 1.0f32)
                };
                self.expect(&Token::RParen)?;
                Ok(Expr::RandomStream { count: Box::new(count), lo, hi })
            }
            Token::Cache => {
                self.advance(); // consume "cache"
                self.expect(&Token::LParen)?;
                let key = match self.peek().clone() {
                    Token::StringLit(s) => { self.advance(); s }
                    _ => return Err(self.error("cache(): expected string key")),
                };
                self.expect(&Token::RParen)?;
                // Inner expression is the full RHS (may include |> pipes)
                let inner = self.parse_expr()?;
                Ok(Expr::Cache { key, inner: Box::new(inner) })
            }
            Token::Ident(name) => {
                self.advance();
                Ok(Expr::Ref { name })
            }
            _ => Err(self.error("expected tap(), random_stream(), cache(), or stream reference")),
        }
    }

    // ── Scalar expression parser (10-level precedence) ────────────
    //
    // Precedence (lowest to highest):
    //   1. if/then/else
    //   2. || (boolean or)
    //   3. && (boolean and)
    //   4. comparison (< > <= >= == !=)
    //   5. | (bitwise or)
    //   6. ^ (bitwise xor)
    //   7. & (bitwise and)
    //   8. << >> (shifts)
    //   9. additive (+ -)
    //  10. multiplicative (* /)
    //  11. atoms (literals, refs, reduce calls, parens, true, false)

    /// scalar_expr → if_expr | or_expr
    fn parse_scalar_expr(&mut self) -> Result<ScalarExpr, ParseError> {
        if matches!(self.peek(), Token::If) {
            self.parse_if_expr()
        } else {
            self.parse_or()
        }
    }

    /// if_expr → "if" or_expr "then" scalar_expr "else" scalar_expr
    fn parse_if_expr(&mut self) -> Result<ScalarExpr, ParseError> {
        self.advance(); // consume "if"
        let condition = self.parse_or()?;
        self.expect(&Token::Then)?;
        let then_expr = self.parse_scalar_expr()?;
        self.expect(&Token::Else)?;
        let else_expr = self.parse_scalar_expr()?;
        Ok(ScalarExpr::If {
            condition: Box::new(condition),
            then_expr: Box::new(then_expr),
            else_expr: Box::new(else_expr),
        })
    }

    /// or_expr → and_expr ("||" and_expr)*
    fn parse_or(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Token::Or) {
            self.advance();
            let right = self.parse_and()?;
            left = ScalarExpr::Or {
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// and_expr → comparison ("&&" comparison)*
    fn parse_and(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_comparison()?;
        while matches!(self.peek(), Token::And) {
            self.advance();
            let right = self.parse_comparison()?;
            left = ScalarExpr::And {
                left: Box::new(left),
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// comparison → bitor (("<" | ">" | "<=" | ">=" | "==" | "!=") bitor)?
    fn parse_comparison(&mut self) -> Result<ScalarExpr, ParseError> {
        let left = self.parse_bitor()?;
        let op = match self.peek() {
            Token::Less => CompareOp::Less,
            Token::Greater => CompareOp::Greater,
            Token::LessEqual => CompareOp::LessEqual,
            Token::GreaterEqual => CompareOp::GreaterEqual,
            Token::EqualEqual => CompareOp::Equal,
            Token::NotEqual => CompareOp::NotEqual,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.parse_bitor()?;
        Ok(ScalarExpr::Compare {
            left: Box::new(left),
            op,
            right: Box::new(right),
        })
    }

    /// bitor → bitxor ('|' bitxor)*
    fn parse_bitor(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_bitxor()?;
        while matches!(self.peek(), Token::BitOr) {
            self.advance();
            let right = self.parse_bitxor()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op: BinOp::BitOr,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// bitxor → bitand ('^' bitand)*
    fn parse_bitxor(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_bitand()?;
        while matches!(self.peek(), Token::BitXor) {
            self.advance();
            let right = self.parse_bitand()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op: BinOp::BitXor,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// bitand → shift ('&' shift)*
    fn parse_bitand(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_shift()?;
        while matches!(self.peek(), Token::BitAnd) {
            self.advance();
            let right = self.parse_shift()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op: BinOp::BitAnd,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// shift → additive (('<<' | '>>') additive)*
    fn parse_shift(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_additive()?;
        loop {
            let op = match self.peek() {
                Token::Shl => BinOp::Shl,
                Token::Shr => BinOp::Shr,
                _ => break,
            };
            self.advance();
            let right = self.parse_additive()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }
        Ok(left)
    }

    /// additive → multiplicative (('+' | '-') multiplicative)*
    fn parse_additive(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_multiplicative()?;

        loop {
            let op = match self.peek() {
                Token::Plus => BinOp::Add,
                Token::Minus => BinOp::Sub,
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// multiplicative → scalar_atom (('*' | '/') scalar_atom)*
    fn parse_multiplicative(&mut self) -> Result<ScalarExpr, ParseError> {
        let mut left = self.parse_scalar_atom()?;

        loop {
            let op = match self.peek() {
                Token::Star => BinOp::Mul,
                Token::Slash => BinOp::Div,
                Token::Percent => BinOp::Mod,
                _ => break,
            };
            self.advance();
            let right = self.parse_scalar_atom()?;
            left = ScalarExpr::BinOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    /// `extern "<library>" { fn name(params) -> type ... }`
    fn parse_extern_block(&mut self) -> Result<Statement, ParseError> {
        let (line, col) = self.current_span();
        let span = Span { line, col };
        self.advance(); // consume 'extern'

        // Expect string literal for library name
        let library = match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); s }
            _ => return Err(self.error("expected library name string after 'extern'")),
        };

        // Skip optional newlines before '{'
        while matches!(self.peek(), Token::Newline) { self.advance(); }

        // Expect '{'
        if !matches!(self.peek(), Token::LBrace) {
            return Err(self.error("expected '{' after extern library name"));
        }
        self.advance(); // consume '{'

        let mut functions: Vec<ExternFn> = Vec::new();

        // Parse function declarations until '}'
        loop {
            // Skip newlines
            while matches!(self.peek(), Token::Newline) { self.advance(); }
            if matches!(self.peek(), Token::RBrace | Token::Eof) { break; }

            // Expect 'fn'
            if !matches!(self.peek(), Token::Fn) {
                return Err(self.error("expected 'fn' inside extern block"));
            }
            self.advance(); // consume 'fn'

            // Function name
            let fn_name = match self.peek().clone() {
                Token::Ident(s) => { self.advance(); s }
                _ => return Err(self.error("expected function name after 'fn' in extern block")),
            };

            // '('
            if !matches!(self.peek(), Token::LParen) {
                return Err(self.error("expected '(' after extern fn name"));
            }
            self.advance(); // consume '('

            // Parameters
            let params = self.parse_extern_params()?;

            // ')'
            if !matches!(self.peek(), Token::RParen) {
                return Err(self.error("expected ')' after extern fn parameters"));
            }
            self.advance(); // consume ')'

            // Optional '-> type'
            let return_type = if matches!(self.peek(), Token::Arrow) {
                self.advance(); // consume '->'
                match self.peek().clone() {
                    Token::Ident(s) => { self.advance(); Some(s) }
                    _ => return Err(self.error("expected return type identifier after '->'"))
                }
            } else {
                None
            };

            // Expect newline (or '}')
            while matches!(self.peek(), Token::Newline) { self.advance(); }

            functions.push(ExternFn { name: fn_name, params, return_type });
        }

        // Expect '}'
        if !matches!(self.peek(), Token::RBrace) {
            return Err(self.error("expected '}' to close extern block"));
        }
        self.advance(); // consume '}'

        // Expect newline or EOF
        if !matches!(self.peek(), Token::Newline | Token::Eof) {
            return Err(self.error("expected newline after extern block"));
        }
        while matches!(self.peek(), Token::Newline) { self.advance(); }

        Ok(Statement::ExternBlock { library, functions, span })
    }

    /// Parse comma-separated extern function parameters: `name: type, ...`
    fn parse_extern_params(&mut self) -> Result<Vec<ExternParam>, ParseError> {
        let mut params = Vec::new();
        if matches!(self.peek(), Token::RParen) {
            return Ok(params);
        }
        params.push(self.parse_extern_param()?);
        while matches!(self.peek(), Token::Comma) {
            self.advance(); // consume ','
            if matches!(self.peek(), Token::RParen) { break; } // trailing comma ok
            params.push(self.parse_extern_param()?);
        }
        Ok(params)
    }

    /// Parse a single extern parameter: `name: type` or `type` (unnamed).
    fn parse_extern_param(&mut self) -> Result<ExternParam, ParseError> {
        let first = match self.peek().clone() {
            Token::Ident(s) => { self.advance(); s }
            _ => return Err(self.error("expected parameter name or type in extern fn")),
        };
        if matches!(self.peek(), Token::Colon) {
            self.advance(); // consume ':'
            let type_name = match self.peek().clone() {
                Token::Ident(s) => { self.advance(); s }
                _ => return Err(self.error("expected type name after ':' in extern fn parameter")),
            };
            Ok(ExternParam { name: first, type_name })
        } else {
            // Unnamed param — use type name as both name and type
            Ok(ExternParam { name: format!("_{}", first), type_name: first })
        }
    }

    fn parse_scalar_atom(&mut self) -> Result<ScalarExpr, ParseError> {
        match self.peek().clone() {
            Token::NumberLit(n) => {
                self.advance();
                Ok(ScalarExpr::Literal(n))
            }
            Token::IntLit(n) => {
                self.advance();
                Ok(ScalarExpr::IntLiteral(n))
            }
            Token::True => {
                self.advance();
                Ok(ScalarExpr::Bool(true))
            }
            Token::False => {
                self.advance();
                Ok(ScalarExpr::Bool(false))
            }
            Token::None => {
                self.advance();
                Ok(ScalarExpr::NoneLiteral)
            }
            Token::LParen => {
                self.advance(); // consume '('
                let expr = self.parse_scalar_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::LBracket => {
                // Array literal expression: [1, 2, 3]
                self.advance(); // consume '['
                let mut elements = Vec::new();
                if !matches!(self.peek(), Token::RBracket) {
                    elements.push(self.parse_scalar_expr()?);
                    while matches!(self.peek(), Token::Comma) {
                        self.advance(); // consume ','
                        if matches!(self.peek(), Token::RBracket) { break; } // trailing comma
                        elements.push(self.parse_scalar_expr()?);
                    }
                }
                self.expect(&Token::RBracket)?;
                Ok(ScalarExpr::ArrayLiteral(elements))
            }
            Token::Ident(name) => {
                self.advance();
                if matches!(self.peek(), Token::LParen) {
                    // Reduce ops on streams: min(stream), max(stream), sum(stream), count(stream)
                    // Also work as regular functions: sum([1,2,3]), min(a, b), etc.
                    if matches!(name.as_str(), "min" | "max" | "sum" | "count") {
                        // Try stream reduce: single ident arg followed by ')'
                        let next_tok = self.tokens.get(self.pos + 1).map(|t| &t.token).unwrap_or(&Token::Eof);
                        if matches!(next_tok, Token::Ident(_)) {
                            let saved = self.pos;
                            self.advance(); // consume '('
                            if let Token::Ident(s) = self.peek().clone() {
                                self.advance();
                                if matches!(self.peek(), Token::RParen) {
                                    self.advance(); // consume ')'
                                    return Ok(ScalarExpr::Reduce { op: name, stream: s });
                                }
                            }
                            // Not a simple stream reduce — backtrack
                            self.pos = saved;
                        }
                        // Fall through to regular function call
                    }
                    {
                        // Scalar function call: abs(x), sqrt(range), pow(x, 2.0)
                        self.advance(); // consume '('
                        let mut args = Vec::new();
                        if !matches!(self.peek(), Token::RParen) {
                            args.push(self.parse_scalar_expr()?);
                            while matches!(self.peek(), Token::Comma) {
                                self.advance(); // consume ','
                                args.push(self.parse_scalar_expr()?);
                            }
                        }
                        self.expect(&Token::RParen)?;
                        Ok(ScalarExpr::FnCall { name, args })
                    }
                } else if matches!(self.peek(), Token::Dot) {
                    // Dotted access: v.x (field), module.fn(args) (prefixed call), module.name (prefixed ref)
                    self.advance(); // consume '.'
                    match self.peek().clone() {
                        Token::Ident(component) => {
                            self.advance();
                            let dotted = format!("{}.{}", name, component);
                            if matches!(self.peek(), Token::LParen) {
                                // Dotted function call: module.fn(args)
                                self.advance(); // consume '('
                                let mut args = Vec::new();
                                if !matches!(self.peek(), Token::RParen) {
                                    args.push(self.parse_scalar_expr()?);
                                    while matches!(self.peek(), Token::Comma) {
                                        self.advance(); // consume ','
                                        args.push(self.parse_scalar_expr()?);
                                    }
                                }
                                self.expect(&Token::RParen)?;
                                Ok(ScalarExpr::FnCall { name: dotted, args })
                            } else {
                                Ok(ScalarExpr::Ref(dotted))
                            }
                        }
                        _ => Err(self.error("expected component name after '.'")),
                    }
                } else if matches!(self.peek(), Token::LBracket) {
                    // Array index access: arr[expr]
                    self.advance(); // consume '['
                    let index = self.parse_scalar_expr()?;
                    self.expect(&Token::RBracket)?;
                    Ok(ScalarExpr::Index { array: name, index: Box::new(index) })
                } else {
                    Ok(ScalarExpr::Ref(name))
                }
            }
            Token::StringLit(s) => {
                self.advance();
                Ok(ScalarExpr::StringLiteral(s))
            }
            Token::Fn => {
                self.advance(); // consume 'fn'
                self.expect(&Token::LParen)?;
                let mut params = Vec::new();
                if !matches!(self.peek(), Token::RParen) {
                    match self.peek().clone() {
                        Token::Ident(p) => { self.advance(); params.push(p); }
                        _ => return Err(self.error("expected parameter name in lambda")),
                    }
                    while matches!(self.peek(), Token::Comma) {
                        self.advance(); // consume ','
                        match self.peek().clone() {
                            Token::Ident(p) => { self.advance(); params.push(p); }
                            _ => return Err(self.error("expected parameter name in lambda")),
                        }
                    }
                }
                self.expect(&Token::RParen)?;
                let body = self.parse_scalar_expr()?;
                self.expect(&Token::End)?;
                Ok(ScalarExpr::Lambda { params, body: Box::new(body) })
            }
            _ => Err(self.error("expected number, string, identifier, 'true', 'false', 'fn', or '(' in scalar expression")),
        }
    }
}

/// Parse a print string into interpolation segments.
///
/// Supports `{name}` for scalar references and `{name:.N}` for precision.
/// Literal `{{` produces a single `{` in output.
fn parse_interpolation(s: &str) -> Result<Vec<PrintSegment>, String> {
    let mut segments = Vec::new();
    let mut literal = String::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '{' {
            // Escaped brace: {{ -> literal {
            if i + 1 < chars.len() && chars[i + 1] == '{' {
                literal.push('{');
                i += 2;
                continue;
            }

            // Flush any preceding literal
            if !literal.is_empty() {
                segments.push(PrintSegment::Literal(literal.clone()));
                literal.clear();
            }

            // Find closing brace
            i += 1; // skip '{'
            let start = i;
            while i < chars.len() && chars[i] != '}' {
                i += 1;
            }
            if i >= chars.len() {
                return Err(format!("unclosed '{{' in print string"));
            }
            let content: String = chars[start..i].iter().collect();
            i += 1; // skip '}'

            if content.is_empty() {
                return Err("empty interpolation '{}' in print string".into());
            }

            // Parse optional precision: "name:.N"
            if let Some(dot_pos) = content.find(":.") {
                let name = content[..dot_pos].to_string();
                let prec_str = &content[dot_pos + 2..];
                let precision: usize = prec_str.parse()
                    .map_err(|_| format!("invalid precision '{}' in '{{{}}}'", prec_str, content))?;
                segments.push(PrintSegment::Scalar { name, precision: Some(precision) });
            } else {
                segments.push(PrintSegment::Scalar { name: content, precision: None });
            }
        } else if chars[i] == '}' {
            // Escaped brace: }} -> literal }
            if i + 1 < chars.len() && chars[i + 1] == '}' {
                literal.push('}');
                i += 2;
                continue;
            }
            return Err("unmatched '}' in print string".into());
        } else {
            literal.push(chars[i]);
            i += 1;
        }
    }

    // Flush final literal
    if !literal.is_empty() {
        segments.push(PrintSegment::Literal(literal));
    }

    // If no segments at all, push empty literal
    if segments.is_empty() {
        segments.push(PrintSegment::Literal(String::new()));
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::*;

    #[test]
    fn test_parse_tap() {
        let prog = parse("stream x = tap(\"a.csv\")\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::StreamDecl { name, expr } => {
                assert_eq!(name, "x");
                match expr {
                    Expr::Tap { path } => assert_eq!(path, "a.csv"),
                    _ => panic!("expected Tap"),
                }
            }
            _ => panic!("expected StreamDecl"),
        }
    }

    #[test]
    fn test_parse_pipe() {
        let prog = parse("stream y = x |> multiply(2.0)\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::StreamDecl { name, expr } => {
                assert_eq!(name, "y");
                match expr {
                    Expr::Pipe { input, stages } => {
                        assert!(matches!(input.as_ref(), Expr::Ref { name } if name == "x"));
                        assert_eq!(stages.len(), 1);
                        assert_eq!(stages[0].operation, "multiply");
                        assert!(matches!(&stages[0].args[0], Arg::Literal(n) if (*n - 2.0).abs() < 1e-10));
                    }
                    _ => panic!("expected Pipe"),
                }
            }
            _ => panic!("expected StreamDecl"),
        }
    }

    #[test]
    fn test_parse_emit() {
        let prog = parse("emit(y, \"out.csv\")\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::Emit { expr, path } => {
                assert!(matches!(expr, Expr::Ref { name } if name == "y"));
                assert_eq!(path, "out.csv");
            }
            _ => panic!("expected Emit"),
        }
    }

    #[test]
    fn test_parse_multi_stage() {
        let prog = parse("stream r = x |> add(1.0) |> sqrt()\n").unwrap();
        match &prog.statements[0].0 {
            Statement::StreamDecl { expr, .. } => match expr {
                Expr::Pipe { stages, .. } => {
                    assert_eq!(stages.len(), 2);
                    assert_eq!(stages[0].operation, "add");
                    assert!(matches!(&stages[0].args[0], Arg::Literal(n) if (*n - 1.0).abs() < 1e-10));
                    assert_eq!(stages[1].operation, "sqrt");
                    assert!(stages[1].args.is_empty());
                }
                _ => panic!("expected Pipe"),
            },
            _ => panic!("expected StreamDecl"),
        }
    }

    #[test]
    fn test_parse_full_program() {
        let source = r#"stream data = tap("input.csv")
stream result = data |> multiply(2.0)
emit(result, "output.csv")
"#;
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 3);
        assert!(matches!(&prog.statements[0].0, Statement::StreamDecl { .. }));
        assert!(matches!(&prog.statements[1].0, Statement::StreamDecl { .. }));
        assert!(matches!(&prog.statements[2].0, Statement::Emit { .. }));
    }

    #[test]
    fn test_reject_missing_equals() {
        let result = parse("stream x tap(\"a.csv\")\n");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("expected"));
        assert!(err.line > 0);
    }

    #[test]
    fn test_reject_unknown_token() {
        let result = parse("stream x = @bad\n");
        assert!(result.is_err());
    }

    #[test]
    fn test_reject_tap_no_string() {
        let result = parse("stream x = tap()\n");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("expected"));
    }

    // ── Phase 3 parser tests ────────────────────────────────────────

    #[test]
    fn test_parse_let_reduce() {
        let prog = parse("let mn = min(data)\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "mn");
                match value {
                    ScalarExpr::Reduce { op, stream } => {
                        assert_eq!(op, "min");
                        assert_eq!(stream, "data");
                    }
                    _ => panic!("expected Reduce, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_let_binop() {
        let prog = parse("let range = mx - mn\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "range");
                match value {
                    ScalarExpr::BinOp { left, op, right } => {
                        assert!(matches!(left.as_ref(), ScalarExpr::Ref(s) if s == "mx"));
                        assert_eq!(*op, BinOp::Sub);
                        assert!(matches!(right.as_ref(), ScalarExpr::Ref(s) if s == "mn"));
                    }
                    _ => panic!("expected BinOp, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_let_precedence() {
        // `2.0 * alpha + 1.0` → Add(Mul(2.0, alpha), 1.0)
        let prog = parse("let x = 2.0 * alpha + 1.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => match value {
                ScalarExpr::BinOp { left, op, right } => {
                    assert_eq!(*op, BinOp::Add);
                    // left should be Mul(2.0, alpha)
                    match left.as_ref() {
                        ScalarExpr::BinOp { left: ll, op: lop, right: lr } => {
                            assert_eq!(*lop, BinOp::Mul);
                            assert!(matches!(ll.as_ref(), ScalarExpr::Literal(n) if (*n - 2.0).abs() < 1e-10));
                            assert!(matches!(lr.as_ref(), ScalarExpr::Ref(s) if s == "alpha"));
                        }
                        _ => panic!("expected Mul in left"),
                    }
                    assert!(matches!(right.as_ref(), ScalarExpr::Literal(n) if (*n - 1.0).abs() < 1e-10));
                }
                _ => panic!("expected BinOp(Add), got {:?}", value),
            },
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_pipe_with_ref_arg() {
        let prog = parse("stream r = data |> subtract(mn)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::StreamDecl { expr, .. } => match expr {
                Expr::Pipe { stages, .. } => {
                    assert_eq!(stages[0].operation, "subtract");
                    assert!(matches!(&stages[0].args[0], Arg::Ref(s) if s == "mn"));
                }
                _ => panic!("expected Pipe"),
            },
            _ => panic!("expected StreamDecl"),
        }
    }

    #[test]
    fn test_parse_print() {
        let prog = parse("print(\"hello world\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::Print { segments } => {
                assert_eq!(segments.len(), 1);
                match &segments[0] {
                    PrintSegment::Literal(s) => assert_eq!(s, "hello world"),
                    _ => panic!("expected Literal segment"),
                }
            }
            _ => panic!("expected Print"),
        }
    }

    #[test]
    fn test_parse_print_interpolation() {
        let prog = parse("print(\"min={mn}, max={mx}\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::Print { segments } => {
                assert_eq!(segments.len(), 4);
                match &segments[0] { PrintSegment::Literal(s) => assert_eq!(s, "min="), _ => panic!() }
                match &segments[1] { PrintSegment::Scalar { name, precision } => { assert_eq!(name, "mn"); assert!(precision.is_none()); }, _ => panic!() }
                match &segments[2] { PrintSegment::Literal(s) => assert_eq!(s, ", max="), _ => panic!() }
                match &segments[3] { PrintSegment::Scalar { name, precision } => { assert_eq!(name, "mx"); assert!(precision.is_none()); }, _ => panic!() }
            }
            _ => panic!("expected Print"),
        }
    }

    #[test]
    fn test_parse_print_precision() {
        let prog = parse("print(\"value={x:.4}\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::Print { segments } => {
                assert_eq!(segments.len(), 2);
                match &segments[1] {
                    PrintSegment::Scalar { name, precision } => {
                        assert_eq!(name, "x");
                        assert_eq!(*precision, Some(4));
                    }
                    _ => panic!("expected Scalar segment"),
                }
            }
            _ => panic!("expected Print"),
        }
    }

    #[test]
    fn test_parse_print_escaped_braces() {
        let prog = parse("print(\"use {{braces}}\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::Print { segments } => {
                assert_eq!(segments.len(), 1);
                match &segments[0] { PrintSegment::Literal(s) => assert_eq!(s, "use {braces}"), _ => panic!() }
            }
            _ => panic!("expected Print"),
        }
    }

    #[test]
    fn test_parse_print_scalar_only() {
        let prog = parse("print(\"{result}\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::Print { segments } => {
                assert_eq!(segments.len(), 1);
                match &segments[0] { PrintSegment::Scalar { name, .. } => assert_eq!(name, "result"), _ => panic!() }
            }
            _ => panic!("expected Print"),
        }
    }

    #[test]
    fn test_reject_let_missing_value() {
        let result = parse("let x =\n");
        assert!(result.is_err());
    }

    // ── Phase 4 parser tests: fn and use ──────────────────────────────

    #[test]
    fn test_parse_fn_decl_same_line() {
        let prog = parse("fn brightness(amount): add(amount) |> clamp(0.0, 255.0)\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::FnDecl { name, params, body } => {
                assert_eq!(name, "brightness");
                assert_eq!(params, &["amount"]);
                assert_eq!(body.len(), 2);
                assert_eq!(body[0].operation, "add");
                assert!(matches!(&body[0].args[0], Arg::Ref(s) if s == "amount"));
                assert_eq!(body[1].operation, "clamp");
                assert_eq!(body[1].args.len(), 2);
            }
            _ => panic!("expected FnDecl"),
        }
    }

    #[test]
    fn test_parse_fn_decl_next_line() {
        let prog = parse("fn gamma(g):\n  divide(255.0) |> pow(g) |> multiply(255.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::FnDecl { name, params, body } => {
                assert_eq!(name, "gamma");
                assert_eq!(params, &["g"]);
                assert_eq!(body.len(), 3);
                assert_eq!(body[0].operation, "divide");
                assert_eq!(body[1].operation, "pow");
                assert!(matches!(&body[1].args[0], Arg::Ref(s) if s == "g"));
                assert_eq!(body[2].operation, "multiply");
            }
            _ => panic!("expected FnDecl"),
        }
    }

    #[test]
    fn test_parse_fn_no_params() {
        let prog = parse("fn invert(): negate() |> add(255.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::FnDecl { name, params, body } => {
                assert_eq!(name, "invert");
                assert!(params.is_empty());
                assert_eq!(body.len(), 2);
            }
            _ => panic!("expected FnDecl"),
        }
    }

    #[test]
    fn test_parse_use_decl() {
        let prog = parse("use filters\n").unwrap();
        assert_eq!(prog.statements.len(), 1);
        match &prog.statements[0].0 {
            Statement::UseDecl { module } => assert_eq!(module, "filters"),
            _ => panic!("expected UseDecl"),
        }
    }

    #[test]
    fn test_parse_dotted_stage_call() {
        let prog = parse("stream r = data |> filters.brightness(20.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::StreamDecl { expr, .. } => match expr {
                Expr::Pipe { stages, .. } => {
                    assert_eq!(stages[0].operation, "filters.brightness");
                    assert!(matches!(&stages[0].args[0], Arg::Literal(n) if (*n - 20.0).abs() < 1e-10));
                }
                _ => panic!("expected Pipe"),
            },
            _ => panic!("expected StreamDecl"),
        }
    }

    #[test]
    fn test_parse_fn_and_call() {
        let source = r#"fn double(): multiply(2.0)
stream data = tap("in.csv")
stream result = data |> double()
"#;
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 3);
        assert!(matches!(&prog.statements[0].0, Statement::FnDecl { .. }));
        assert!(matches!(&prog.statements[1].0, Statement::StreamDecl { .. }));
        assert!(matches!(&prog.statements[2].0, Statement::StreamDecl { .. }));
    }

    // ── Phase 8 parser tests: if/then/else, comparisons, booleans ────

    #[test]
    fn test_parse_let_comparison() {
        let prog = parse("let x = a > 5.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                match value {
                    ScalarExpr::Compare { left, op, right } => {
                        assert!(matches!(left.as_ref(), ScalarExpr::Ref(s) if s == "a"));
                        assert_eq!(*op, CompareOp::Greater);
                        assert!(matches!(right.as_ref(), ScalarExpr::Literal(n) if (*n - 5.0).abs() < 1e-10));
                    }
                    _ => panic!("expected Compare, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_let_if_then_else() {
        let prog = parse("let x = if a > 0.0 then 1.0 else 0.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                match value {
                    ScalarExpr::If { condition, then_expr, else_expr } => {
                        assert!(matches!(condition.as_ref(), ScalarExpr::Compare { .. }));
                        assert!(matches!(then_expr.as_ref(), ScalarExpr::Literal(n) if (*n - 1.0).abs() < 1e-10));
                        assert!(matches!(else_expr.as_ref(), ScalarExpr::Literal(n) if (*n - 0.0).abs() < 1e-10));
                    }
                    _ => panic!("expected If, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_nested_if() {
        // if a > 0 then 1.0 else if a < -1.0 then -1.0 else 0.0
        let prog = parse("let x = if a > 0.0 then 1.0 else if a < -1.0 then -1.0 else 0.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::If { else_expr, .. } => {
                        // else branch is another if
                        assert!(matches!(else_expr.as_ref(), ScalarExpr::If { .. }));
                    }
                    _ => panic!("expected If"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_boolean_and_or() {
        let prog = parse("let x = a > 0.0 && b < 10.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                assert!(matches!(value, ScalarExpr::And { .. }));
            }
            _ => panic!("expected LetDecl"),
        }

        let prog = parse("let x = a > 0.0 || b > 0.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                assert!(matches!(value, ScalarExpr::Or { .. }));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_boolean_precedence() {
        // `a > 0 && b > 0 || c > 0` should parse as `(a>0 && b>0) || c>0`
        let prog = parse("let x = a > 0.0 && b > 0.0 || c > 0.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::Or { left, right } => {
                        assert!(matches!(left.as_ref(), ScalarExpr::And { .. }));
                        assert!(matches!(right.as_ref(), ScalarExpr::Compare { .. }));
                    }
                    _ => panic!("expected Or, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_true_false() {
        let prog = parse("let x = true\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                assert!(matches!(value, ScalarExpr::Bool(true)));
            }
            _ => panic!("expected LetDecl"),
        }

        let prog = parse("let x = false\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                assert!(matches!(value, ScalarExpr::Bool(false)));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_all_comparison_ops() {
        for (src, expected_op) in &[
            ("a < b", CompareOp::Less),
            ("a > b", CompareOp::Greater),
            ("a <= b", CompareOp::LessEqual),
            ("a >= b", CompareOp::GreaterEqual),
            ("a == b", CompareOp::Equal),
            ("a != b", CompareOp::NotEqual),
        ] {
            let prog = parse(&format!("let x = {}\n", src)).unwrap();
            match &prog.statements[0].0 {
                Statement::LetDecl { value, .. } => {
                    match value {
                        ScalarExpr::Compare { op, .. } => assert_eq!(op, expected_op),
                        _ => panic!("expected Compare for '{}'", src),
                    }
                }
                _ => panic!("expected LetDecl"),
            }
        }
    }

    #[test]
    fn test_parse_parenthesized_scalar() {
        let prog = parse("let x = (a + b) * c\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::BinOp { left, op, .. } => {
                        assert_eq!(*op, BinOp::Mul);
                        // left should be Add(a, b) due to parens
                        assert!(matches!(left.as_ref(), ScalarExpr::BinOp { op: BinOp::Add, .. }));
                    }
                    _ => panic!("expected BinOp(Mul)"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    // ── Phase 12 parser tests: scalar functions + count ──────────────

    #[test]
    fn test_parse_scalar_fn_call_1arg() {
        let prog = parse("let x = abs(val)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "abs");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], ScalarExpr::Ref(s) if s == "val"));
                    }
                    _ => panic!("expected FnCall, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_call_2args() {
        let prog = parse("let x = pow(base, 2.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "pow");
                        assert_eq!(args.len(), 2);
                        assert!(matches!(&args[0], ScalarExpr::Ref(s) if s == "base"));
                        assert!(matches!(&args[1], ScalarExpr::Literal(n) if (*n - 2.0).abs() < 1e-10));
                    }
                    _ => panic!("expected FnCall, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_call_3args() {
        let prog = parse("let x = clamp(val, 0.0, 255.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "clamp");
                        assert_eq!(args.len(), 3);
                    }
                    _ => panic!("expected FnCall, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_nested() {
        // sqrt(abs(x)) — nested function calls
        let prog = parse("let x = sqrt(abs(val))\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "sqrt");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], ScalarExpr::FnCall { name, .. } if name == "abs"));
                    }
                    _ => panic!("expected FnCall, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_in_expr() {
        // abs(x) + sqrt(y) — function calls in arithmetic
        let prog = parse("let z = abs(x) + sqrt(y)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::BinOp { left, op, right } => {
                        assert_eq!(*op, BinOp::Add);
                        assert!(matches!(left.as_ref(), ScalarExpr::FnCall { name, .. } if name == "abs"));
                        assert!(matches!(right.as_ref(), ScalarExpr::FnCall { name, .. } if name == "sqrt"));
                    }
                    _ => panic!("expected BinOp, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_count_reduce() {
        let prog = parse("let n = count(data)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "n");
                match value {
                    ScalarExpr::Reduce { op, stream } => {
                        assert_eq!(op, "count");
                        assert_eq!(stream, "data");
                    }
                    _ => panic!("expected Reduce, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_min_still_reduce() {
        // Ensure min/max/sum still parse as Reduce (not FnCall)
        let prog = parse("let mn = min(data)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                assert!(matches!(value, ScalarExpr::Reduce { op, .. } if op == "min"));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    // ── Phase 14 parser tests: vec types + dotted refs ─────────────────

    #[test]
    fn test_parse_vec3_constructor() {
        let prog = parse("let pos = vec3(1.0, 2.0, 3.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "pos");
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "vec3");
                        assert_eq!(args.len(), 3);
                    }
                    _ => panic!("expected FnCall(vec3), got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_vec2_constructor() {
        let prog = parse("let uv = vec2(0.5, 0.5)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "uv");
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "vec2");
                        assert_eq!(args.len(), 2);
                    }
                    _ => panic!("expected FnCall(vec2)"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_vec4_constructor() {
        let prog = parse("let color = vec4(1.0, 0.0, 0.0, 1.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "color");
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "vec4");
                        assert_eq!(args.len(), 4);
                    }
                    _ => panic!("expected FnCall(vec4)"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_dotted_scalar_ref() {
        let prog = parse("let x = pos.x\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(value, ScalarExpr::Ref(s) if s == "pos.x"));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_dotted_ref_in_expr() {
        let prog = parse("let dist = sqrt(pos.x * pos.x + pos.y * pos.y)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "dist");
                assert!(matches!(value, ScalarExpr::FnCall { name, .. } if name == "sqrt"));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_vec3_with_scalar_args() {
        let prog = parse("let v = vec3(a + 1.0, b * 2.0, c)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::FnCall { name, args } => {
                        assert_eq!(name, "vec3");
                        assert_eq!(args.len(), 3);
                        // First arg should be BinOp(Add)
                        assert!(matches!(&args[0], ScalarExpr::BinOp { op: BinOp::Add, .. }));
                    }
                    _ => panic!("expected FnCall(vec3)"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_struct_def() {
        let prog = parse("struct Point(x, y, z)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::StructDef { name, fields } => {
                assert_eq!(name, "Point");
                assert_eq!(fields, &["x", "y", "z"]);
            }
            _ => panic!("expected StructDef"),
        }
    }

    #[test]
    fn test_parse_struct_def_two_fields() {
        let prog = parse("struct UV(u, v)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::StructDef { name, fields } => {
                assert_eq!(name, "UV");
                assert_eq!(fields, &["u", "v"]);
            }
            _ => panic!("expected StructDef"),
        }
    }

    #[test]
    fn test_parse_struct_constructor() {
        let prog = parse("struct Entity(x, y, health)\nlet player = Entity(1.0, 2.0, 100.0)\n").unwrap();
        assert_eq!(prog.statements.len(), 2);
        // First statement is StructDef
        assert!(matches!(&prog.statements[0].0, Statement::StructDef { name, .. } if name == "Entity"));
        // Second is LetDecl with FnCall
        match &prog.statements[1].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "player");
                match value {
                    ScalarExpr::FnCall { name: fn_name, args } => {
                        assert_eq!(fn_name, "Entity");
                        assert_eq!(args.len(), 3);
                    }
                    _ => panic!("expected FnCall(Entity)"),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_struct_field_access() {
        // Dotted ref parsing works for struct fields just like vec components
        let prog = parse("let hp = player.health\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "hp");
                assert!(matches!(value, ScalarExpr::Ref(s) if s == "player.health"));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    // ── Phase 16 parser tests: arrays + indexing ─────────────────────────

    #[test]
    fn test_parse_array_decl() {
        let prog = parse("let arr = [1.0, 2.0, 3.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayDecl { name, elements, .. } => {
                assert_eq!(name, "arr");
                assert_eq!(elements.len(), 3);
                assert!(matches!(&elements[0], ScalarExpr::Literal(n) if (*n - 1.0).abs() < 1e-10));
                assert!(matches!(&elements[2], ScalarExpr::Literal(n) if (*n - 3.0).abs() < 1e-10));
            }
            _ => panic!("expected ArrayDecl, got {:?}", prog.statements[0].0),
        }
    }

    #[test]
    fn test_parse_array_decl_empty() {
        let prog = parse("let arr = []\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayDecl { name, elements, .. } => {
                assert_eq!(name, "arr");
                assert!(elements.is_empty());
            }
            _ => panic!("expected ArrayDecl"),
        }
    }

    #[test]
    fn test_parse_array_decl_with_exprs() {
        let prog = parse("let arr = [1.0 + 2.0, x * 3.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayDecl { name, elements, .. } => {
                assert_eq!(name, "arr");
                assert_eq!(elements.len(), 2);
                assert!(matches!(&elements[0], ScalarExpr::BinOp { op: BinOp::Add, .. }));
            }
            _ => panic!("expected ArrayDecl"),
        }
    }

    #[test]
    fn test_parse_array_index() {
        let prog = parse("let x = arr[0.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                match value {
                    ScalarExpr::Index { array, index } => {
                        assert_eq!(array, "arr");
                        assert!(matches!(index.as_ref(), ScalarExpr::Literal(n) if (*n - 0.0).abs() < 1e-10));
                    }
                    _ => panic!("expected Index, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_array_index_expr() {
        let prog = parse("let x = arr[i + 1.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                match value {
                    ScalarExpr::Index { array, index } => {
                        assert_eq!(array, "arr");
                        assert!(matches!(index.as_ref(), ScalarExpr::BinOp { op: BinOp::Add, .. }));
                    }
                    _ => panic!("expected Index, got {:?}", value),
                }
            }
            _ => panic!("expected LetDecl"),
        }
    }

    // ── Phase 17 parser tests: mutable state ────────────────────────────

    #[test]
    fn test_parse_let_mut() {
        let prog = parse("let mut x = 0.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, mutable } => {
                assert_eq!(name, "x");
                assert!(*mutable);
                assert!(matches!(value, ScalarExpr::Literal(n) if (*n - 0.0).abs() < 1e-10));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_let_immutable() {
        let prog = parse("let x = 5.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, mutable, .. } => {
                assert_eq!(name, "x");
                assert!(!*mutable);
            }
            _ => panic!("expected LetDecl"),
        }
    }

    #[test]
    fn test_parse_assign() {
        let prog = parse("let mut x = 0.0\nx = x + 1.0\n").unwrap();
        assert_eq!(prog.statements.len(), 2);
        match &prog.statements[1].0 {
            Statement::Assign { name, value } => {
                assert_eq!(name, "x");
                assert!(matches!(value, ScalarExpr::BinOp { op: BinOp::Add, .. }));
            }
            _ => panic!("expected Assign, got {:?}", prog.statements[1].0),
        }
    }

    #[test]
    fn test_parse_multiple_assigns() {
        let source = "let mut score = 0.0\nscore = score + 10.0\nscore = score + 5.0\n";
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 3);
        assert!(matches!(&prog.statements[0].0, Statement::LetDecl { mutable: true, .. }));
        assert!(matches!(&prog.statements[1].0, Statement::Assign { .. }));
        assert!(matches!(&prog.statements[2].0, Statement::Assign { .. }));
    }

    #[test]
    fn test_parse_let_mut_with_expr() {
        let prog = parse("let a = 1.0\nlet mut x = a + 2.0\n").unwrap();
        assert_eq!(prog.statements.len(), 2);
        match &prog.statements[1].0 {
            Statement::LetDecl { name, mutable, value } => {
                assert_eq!(name, "x");
                assert!(*mutable);
                assert!(matches!(value, ScalarExpr::BinOp { op: BinOp::Add, .. }));
            }
            _ => panic!("expected LetDecl"),
        }
    }

    // ── Phase 19 parser tests: while loops ──────────────────────────────

    #[test]
    fn test_parse_while_simple() {
        let source = "let mut x = 10.0\nwhile x > 0.0\n  x = x - 1.0\nend\n";
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 2);
        match &prog.statements[1].0 {
            Statement::WhileLoop { condition, body } => {
                assert!(matches!(condition, ScalarExpr::Compare { op: CompareOp::Greater, .. }));
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Assign { .. }));
            }
            _ => panic!("expected WhileLoop, got {:?}", prog.statements[1].0),
        }
    }

    #[test]
    fn test_parse_while_multiple_body_stmts() {
        let source = "let mut a = 0.0\nlet mut b = 10.0\nwhile b > 0.0\n  a = a + b\n  b = b - 1.0\nend\n";
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 3);
        match &prog.statements[2].0 {
            Statement::WhileLoop { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(matches!(&body[0].0, Statement::Assign { name, .. } if name == "a"));
                assert!(matches!(&body[1].0, Statement::Assign { name, .. } if name == "b"));
            }
            _ => panic!("expected WhileLoop"),
        }
    }

    #[test]
    fn test_parse_while_with_print() {
        let source = "let mut i = 3.0\nwhile i > 0.0\n  print(\"i={i}\")\n  i = i - 1.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::WhileLoop { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(matches!(&body[0].0, Statement::Print { .. }));
                assert!(matches!(&body[1].0, Statement::Assign { .. }));
            }
            _ => panic!("expected WhileLoop"),
        }
    }

    #[test]
    fn test_parse_while_missing_end() {
        let source = "let mut x = 5.0\nwhile x > 0.0\n  x = x - 1.0\n";
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("end"), "Error: {}", err.message);
    }

    #[test]
    fn test_parse_while_with_let() {
        let source = "let mut x = 0.0\nwhile x < 5.0\n  let y = x * 2.0\n  x = x + 1.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::WhileLoop { body, .. } => {
                assert_eq!(body.len(), 2);
                assert!(matches!(&body[0].0, Statement::LetDecl { name, .. } if name == "y"));
                assert!(matches!(&body[1].0, Statement::Assign { .. }));
            }
            _ => panic!("expected WhileLoop"),
        }
    }

    // ── Phase 20: for-loop parsing ─────────────────────────────────

    #[test]
    fn test_parse_for_simple() {
        let source = "let mut total = 0.0\nfor i in range(0, 5)\n  total = total + i\nend\n";
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 2);
        match &prog.statements[1].0 {
            Statement::ForLoop { var, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Assign { .. }));
            }
            _ => panic!("expected ForLoop, got {:?}", prog.statements[1].0),
        }
    }

    #[test]
    fn test_parse_for_with_let() {
        let source = "for i in range(1, 4)\n  let x = i * 2.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { var, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::LetDecl { name, .. } if name == "x"));
            }
            _ => panic!("expected ForLoop"),
        }
    }

    #[test]
    fn test_parse_for_with_print() {
        let source = "for i in range(0, 3)\n  print(\"i={i}\")\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Print { .. }));
            }
            _ => panic!("expected ForLoop"),
        }
    }

    #[test]
    fn test_parse_for_missing_end() {
        let source = "for i in range(0, 3)\n  let x = 1.0\n";
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("end"), "Error: {}", err.message);
    }

    #[test]
    fn test_parse_for_scalar_range_args() {
        let source = "let n = 10.0\nfor i in range(0, n)\n  let x = i\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::ForLoop { var, body, .. } => {
                assert_eq!(var, "i");
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected ForLoop"),
        }
    }

    // ── Phase 22: Break / Continue ──────────────────────────────

    #[test]
    fn test_parse_break_in_while() {
        let source = "let mut x = 0.0\nwhile x < 10.0\n  break\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::WhileLoop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Break));
            }
            _ => panic!("expected WhileLoop"),
        }
    }

    #[test]
    fn test_parse_continue_in_for() {
        let source = "for i in range(0, 5)\n  continue\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Continue));
            }
            _ => panic!("expected ForLoop"),
        }
    }

    #[test]
    fn test_parse_break_continue_mixed() {
        let source = "for i in range(0, 10)\n  let x = i\n  continue\n  break\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { body, .. } => {
                assert_eq!(body.len(), 3);
                assert!(matches!(&body[1].0, Statement::Continue));
                assert!(matches!(&body[2].0, Statement::Break));
            }
            _ => panic!("expected ForLoop"),
        }
    }

    #[test]
    fn test_parse_break_standalone_error() {
        // break at top level should still parse (semantic error caught later)
        let source = "break\n";
        let prog = parse(source);
        // Parser should accept it — preflight catches misuse
        assert!(prog.is_ok());
        assert!(matches!(&prog.unwrap().statements[0].0, Statement::Break));
    }

    #[test]
    fn test_parse_continue_standalone_error() {
        let source = "continue\n";
        let prog = parse(source);
        assert!(prog.is_ok());
        assert!(matches!(&prog.unwrap().statements[0].0, Statement::Continue));
    }

    // ── Phase 23: If/Elif/Else Statement Blocks ─────────────────

    #[test]
    fn test_parse_if_block_simple() {
        let source = "let x = 1.0\nif x > 0.0\n  let y = 2.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::IfBlock { body, elif_branches, else_body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(elif_branches.is_empty());
                assert!(else_body.is_empty());
            }
            _ => panic!("expected IfBlock"),
        }
    }

    #[test]
    fn test_parse_if_else_block() {
        let source = "let x = 1.0\nif x > 0.0\n  let y = 1.0\nelse\n  let y = 0.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::IfBlock { body, elif_branches, else_body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(elif_branches.is_empty());
                assert_eq!(else_body.len(), 1);
            }
            _ => panic!("expected IfBlock"),
        }
    }

    #[test]
    fn test_parse_if_elif_else() {
        let source = "let x = 1.0\nif x > 10.0\n  let y = 3.0\nelif x > 5.0\n  let y = 2.0\nelif x > 0.0\n  let y = 1.0\nelse\n  let y = 0.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::IfBlock { body, elif_branches, else_body, .. } => {
                assert_eq!(body.len(), 1);
                assert_eq!(elif_branches.len(), 2);
                assert_eq!(else_body.len(), 1);
            }
            _ => panic!("expected IfBlock"),
        }
    }

    #[test]
    fn test_parse_if_block_multi_stmt() {
        let source = "let mut x = 0.0\nif true\n  x = 1.0\n  x = x + 1.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::IfBlock { body, .. } => {
                assert_eq!(body.len(), 2);
            }
            _ => panic!("expected IfBlock"),
        }
    }

    #[test]
    fn test_parse_if_block_missing_end() {
        let source = "if true\n  let x = 1.0\n";
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("end"), "Error: {}", err.message);
    }

    #[test]
    fn test_parse_if_block_in_loop() {
        let source = "for i in range(0, 5)\n  if i > 2.0\n    break\n  end\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::IfBlock { .. }));
            }
            _ => panic!("expected ForLoop"),
        }
    }

    #[test]
    fn test_parse_elif_keyword() {
        // Verify elif is recognized as keyword
        let tokens = lexer::tokenize("elif\n").unwrap();
        assert!(matches!(tokens[0].token, lexer::Token::Elif));
    }

    // ── Phase 24: Scalar functions ──────────────────────────────────

    #[test]
    fn test_parse_return_keyword() {
        let tokens = lexer::tokenize("return\n").unwrap();
        assert!(matches!(tokens[0].token, lexer::Token::Return));
    }

    #[test]
    fn test_parse_scalar_fn_simple() {
        let source = "fn double(x)\n  return x * 2.0\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ScalarFnDecl { name, params, body } => {
                assert_eq!(name, "double");
                assert_eq!(params, &["x"]);
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Return { .. }));
            }
            _ => panic!("expected ScalarFnDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_multi_params() {
        let source = "fn add(a, b)\n  return a + b\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ScalarFnDecl { name, params, body } => {
                assert_eq!(name, "add");
                assert_eq!(params, &["a", "b"]);
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected ScalarFnDecl"),
        }
    }

    #[test]
    fn test_parse_scalar_fn_with_locals() {
        let source = "fn dist(x1, y1, x2, y2)\n  let dx = x2 - x1\n  let dy = y2 - y1\n  return sqrt(dx * dx + dy * dy)\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[0].0 {
            Statement::ScalarFnDecl { name, params, body } => {
                assert_eq!(name, "dist");
                assert_eq!(params.len(), 4);
                assert_eq!(body.len(), 3); // 2 lets + 1 return
            }
            _ => panic!("expected ScalarFnDecl"),
        }
    }

    #[test]
    fn test_parse_pipeline_fn_still_works() {
        // Verify pipeline fn syntax with ':' still parses correctly
        let source = "fn double(x): multiply(2.0)\n";
        let prog = parse(source).unwrap();
        assert!(matches!(&prog.statements[0].0, Statement::FnDecl { .. }));
    }

    #[test]
    fn test_parse_scalar_fn_missing_end() {
        let source = "fn broken(x)\n  return x\n";
        let result = parse(source);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_return_standalone() {
        let source = "return 42.0\n";
        let prog = parse(source).unwrap();
        assert!(matches!(&prog.statements[0].0, Statement::Return { .. }));
    }

    // ── Phase 28: For-Each Loops ──────────────────────────────────

    #[test]
    fn test_parse_foreach_simple() {
        let source = "let arr = [1.0, 2.0, 3.0]\nfor x in arr\n  let y = x\nend\n";
        let prog = parse(source).unwrap();
        assert_eq!(prog.statements.len(), 2);
        match &prog.statements[1].0 {
            Statement::ForEachLoop { var, iterable, body } => {
                assert_eq!(var, "x");
                assert_eq!(iterable, "arr");
                assert_eq!(body.len(), 1);
            }
            _ => panic!("expected ForEachLoop, got {:?}", prog.statements[1].0),
        }
    }

    #[test]
    fn test_parse_foreach_with_print() {
        let source = "let data = [10.0, 20.0]\nfor val in data\n  print(\"{val}\")\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::ForEachLoop { var, iterable, body } => {
                assert_eq!(var, "val");
                assert_eq!(iterable, "data");
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Print { .. }));
            }
            _ => panic!("expected ForEachLoop"),
        }
    }

    #[test]
    fn test_parse_foreach_multi_body() {
        let source = "let arr = [1.0]\nfor x in arr\n  let y = x * 2.0\n  print(\"{y}\")\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::ForEachLoop { body, .. } => {
                assert_eq!(body.len(), 2);
            }
            _ => panic!("expected ForEachLoop"),
        }
    }

    #[test]
    fn test_parse_foreach_missing_end() {
        let source = "let arr = [1.0]\nfor x in arr\n  let y = x\n";
        let result = parse(source);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("end"), "Error: {}", err.message);
    }

    #[test]
    fn test_parse_foreach_with_break() {
        let source = "let arr = [1.0, 2.0]\nfor x in arr\n  break\nend\n";
        let prog = parse(source).unwrap();
        match &prog.statements[1].0 {
            Statement::ForEachLoop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(&body[0].0, Statement::Break));
            }
            _ => panic!("expected ForEachLoop"),
        }
    }

    #[test]
    fn test_parse_for_range_still_works() {
        // Verify existing for-range syntax is unbroken
        let source = "for i in range(0, 5)\n  let x = i\nend\n";
        let prog = parse(source).unwrap();
        assert!(matches!(&prog.statements[0].0, Statement::ForLoop { .. }));
    }

    // ── Phase 29: array mutation parsing ──────────────────────────

    #[test]
    fn test_parse_array_assign() {
        let prog = parse("arr[0] = 5.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayAssign { array, index, value } => {
                assert_eq!(array, "arr");
                assert!(matches!(index, ScalarExpr::IntLiteral(0)));
                assert!(matches!(value, ScalarExpr::Literal(n) if (*n - 5.0).abs() < 1e-10));
            }
            other => panic!("expected ArrayAssign, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_array_assign_expr_index() {
        let prog = parse("arr[i + 1] = x * 2.0\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayAssign { array, index, value } => {
                assert_eq!(array, "arr");
                assert!(matches!(index, ScalarExpr::BinOp { op: BinOp::Add, .. }));
                assert!(matches!(value, ScalarExpr::BinOp { op: BinOp::Mul, .. }));
            }
            other => panic!("expected ArrayAssign, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_push() {
        let prog = parse("push(arr, 5.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayPush { array, value } => {
                assert_eq!(array, "arr");
                assert!(matches!(value, ScalarExpr::Literal(n) if (*n - 5.0).abs() < 1e-10));
            }
            other => panic!("expected ArrayPush, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_push_expr_value() {
        let prog = parse("push(data, x + 1.0)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayPush { array, value } => {
                assert_eq!(array, "data");
                assert!(matches!(value, ScalarExpr::BinOp { op: BinOp::Add, .. }));
            }
            other => panic!("expected ArrayPush, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_mutable_array_decl() {
        let prog = parse("let mut arr = [1.0, 2.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayDecl { name, elements, mutable } => {
                assert_eq!(name, "arr");
                assert_eq!(elements.len(), 2);
                assert!(*mutable);
            }
            other => panic!("expected ArrayDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_immutable_array_decl() {
        let prog = parse("let arr = [1.0]\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ArrayDecl { mutable, .. } => {
                assert!(!*mutable);
            }
            other => panic!("expected ArrayDecl, got {:?}", other),
        }
    }

    // ── Phase 30b: HashMap parsing tests ────────────────────────────

    #[test]
    fn test_parse_map_decl() {
        let prog = parse("let mut m = map()\n").unwrap();
        match &prog.statements[0].0 {
            Statement::MapDecl { name, mutable } => {
                assert_eq!(name, "m");
                assert!(*mutable);
            }
            other => panic!("expected MapDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_map_decl_immutable() {
        let prog = parse("let m = map()\n").unwrap();
        match &prog.statements[0].0 {
            Statement::MapDecl { name, mutable } => {
                assert_eq!(name, "m");
                assert!(!*mutable);
            }
            other => panic!("expected MapDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_map_set() {
        let prog = parse("let mut m = map()\nmap_set(m, \"key\", 42.0)\n").unwrap();
        match &prog.statements[1].0 {
            Statement::MapInsert { map, .. } => {
                assert_eq!(map, "m");
            }
            other => panic!("expected MapInsert, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_map_get_in_let() {
        let prog = parse("let mut m = map()\nlet v = map_get(m, \"key\")\n").unwrap();
        match &prog.statements[1].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "v");
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    assert_eq!(fn_name, "map_get");
                    assert_eq!(args.len(), 2);
                } else {
                    panic!("expected FnCall, got {:?}", value);
                }
            }
            other => panic!("expected LetDecl, got {:?}", other),
        }
    }

    // ── Phase 31: File I/O parser tests ──────────────────────────

    #[test]
    fn test_parse_write_file() {
        let prog = parse("write_file(\"out.txt\", \"data\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::WriteFile { path, content } => {
                assert!(matches!(path, ScalarExpr::StringLiteral(s) if s == "out.txt"));
                assert!(matches!(content, ScalarExpr::StringLiteral(s) if s == "data"));
            }
            other => panic!("expected WriteFile, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_append_file() {
        let prog = parse("append_file(\"log.txt\", \"entry\")\n").unwrap();
        match &prog.statements[0].0 {
            Statement::AppendFile { path, content } => {
                assert!(matches!(path, ScalarExpr::StringLiteral(s) if s == "log.txt"));
                assert!(matches!(content, ScalarExpr::StringLiteral(s) if s == "entry"));
            }
            other => panic!("expected AppendFile, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_write_file_with_expr() {
        let prog = parse("let p = \"out.txt\"\nwrite_file(p, \"data\")\n").unwrap();
        match &prog.statements[1].0 {
            Statement::WriteFile { path, .. } => {
                assert!(matches!(path, ScalarExpr::Ref(s) if s == "p"));
            }
            other => panic!("expected WriteFile, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_write_file_in_loop() {
        let prog = parse("for i in range(0, 3)\n  write_file(\"out.txt\", \"data\")\nend\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ForLoop { body, .. } => {
                assert!(matches!(&body[0].0, Statement::WriteFile { .. }));
            }
            other => panic!("expected ForLoop, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_save_data() {
        let prog = parse("save_data(\"config.od\", settings)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::SaveData { path, map_name } => {
                assert!(matches!(path, ScalarExpr::StringLiteral(s) if s == "config.od"));
                assert_eq!(map_name, "settings");
            }
            other => panic!("expected SaveData, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_save_data_expr_path() {
        let prog = parse("let p = \"out.od\"\nsave_data(p, mymap)\n").unwrap();
        match &prog.statements[1].0 {
            Statement::SaveData { path, map_name } => {
                assert!(matches!(path, ScalarExpr::Ref(s) if s == "p"));
                assert_eq!(map_name, "mymap");
            }
            other => panic!("expected SaveData, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_lambda_single_param() {
        let prog = parse("let x = filter(arr, fn(x) x > 0.0 end)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    assert_eq!(fn_name, "filter");
                    assert_eq!(args.len(), 2);
                    if let ScalarExpr::Lambda { params, .. } = &args[1] {
                        assert_eq!(params, &["x"]);
                    } else {
                        panic!("expected Lambda, got {:?}", args[1]);
                    }
                } else {
                    panic!("expected FnCall, got {:?}", value);
                }
            }
            other => panic!("expected LetDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_lambda_two_params() {
        let prog = parse("let x = reduce(arr, 0.0, fn(acc, x) acc + x end)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { name, value, .. } => {
                assert_eq!(name, "x");
                if let ScalarExpr::FnCall { name: fn_name, args } = value {
                    assert_eq!(fn_name, "reduce");
                    assert_eq!(args.len(), 3);
                    if let ScalarExpr::Lambda { params, .. } = &args[2] {
                        assert_eq!(params, &["acc", "x"]);
                    } else {
                        panic!("expected Lambda, got {:?}", args[2]);
                    }
                } else {
                    panic!("expected FnCall, got {:?}", value);
                }
            }
            other => panic!("expected LetDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_lambda_in_fn_call() {
        let prog = parse("let r = map_each(nums, fn(x) x * 2.0 end)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                if let ScalarExpr::FnCall { name, args } = value {
                    assert_eq!(name, "map_each");
                    assert_eq!(args.len(), 2);
                    assert!(matches!(&args[1], ScalarExpr::Lambda { .. }));
                } else {
                    panic!("expected FnCall");
                }
            }
            other => panic!("expected LetDecl, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_lambda_nested_expr() {
        let prog = parse("let r = map_each(arr, fn(x) abs(x) + 1.0 end)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::LetDecl { value, .. } => {
                if let ScalarExpr::FnCall { name, args } = value {
                    assert_eq!(name, "map_each");
                    if let ScalarExpr::Lambda { params, body } = &args[1] {
                        assert_eq!(params, &["x"]);
                        assert!(matches!(body.as_ref(), ScalarExpr::BinOp { .. }));
                    } else {
                        panic!("expected Lambda");
                    }
                } else {
                    panic!("expected FnCall");
                }
            }
            other => panic!("expected LetDecl, got {:?}", other),
        }
    }

    // ── Phase 39: write_csv ────────────────────────────────────────────

    #[test]
    fn test_parse_write_csv() {
        let prog = parse("write_csv(\"out.csv\", data)\n").unwrap();
        match &prog.statements[0].0 {
            Statement::WriteCsv { path, array_name } => {
                assert!(matches!(path, ScalarExpr::StringLiteral(s) if s == "out.csv"));
                assert_eq!(array_name, "data");
            }
            other => panic!("expected WriteCsv, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_write_csv_expr_path() {
        let prog = parse("let p = \"out.csv\"\nwrite_csv(p, results)\n").unwrap();
        match &prog.statements[1].0 {
            Statement::WriteCsv { path, array_name } => {
                assert!(matches!(path, ScalarExpr::Ref(s) if s == "p"));
                assert_eq!(array_name, "results");
            }
            other => panic!("expected WriteCsv, got {:?}", other),
        }
    }

    // ── Phase 44: extern FFI ──────────────────────────────────────────

    #[test]
    fn test_parse_extern_block_simple() {
        let prog = parse("extern \"mylib\" {\n  fn add(a: i32, b: i32) -> i32\n}\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ExternBlock { library, functions, .. } => {
                assert_eq!(library, "mylib");
                assert_eq!(functions.len(), 1);
                assert_eq!(functions[0].name, "add");
                assert_eq!(functions[0].params.len(), 2);
                assert_eq!(functions[0].params[0].name, "a");
                assert_eq!(functions[0].params[0].type_name, "i32");
                assert_eq!(functions[0].params[1].name, "b");
                assert_eq!(functions[0].params[1].type_name, "i32");
                assert_eq!(functions[0].return_type.as_deref(), Some("i32"));
            }
            other => panic!("expected ExternBlock, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_extern_block_void_return() {
        let prog = parse("extern \"c\" {\n  fn puts(s: ptr)\n}\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ExternBlock { library, functions, .. } => {
                assert_eq!(library, "c");
                assert_eq!(functions[0].name, "puts");
                assert_eq!(functions[0].params[0].type_name, "ptr");
                assert!(functions[0].return_type.is_none());
            }
            other => panic!("expected ExternBlock, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_extern_block_multiple_fns() {
        let src = "extern \"math\" {\n  fn sin(x: f32) -> f32\n  fn cos(x: f32) -> f32\n  fn sqrt(x: f32) -> f32\n}\n";
        let prog = parse(src).unwrap();
        match &prog.statements[0].0 {
            Statement::ExternBlock { library, functions, .. } => {
                assert_eq!(library, "math");
                assert_eq!(functions.len(), 3);
                assert_eq!(functions[0].name, "sin");
                assert_eq!(functions[1].name, "cos");
                assert_eq!(functions[2].name, "sqrt");
            }
            other => panic!("expected ExternBlock, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_extern_block_no_params() {
        let prog = parse("extern \"c\" {\n  fn clock() -> i64\n}\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ExternBlock { functions, .. } => {
                assert_eq!(functions[0].name, "clock");
                assert!(functions[0].params.is_empty());
                assert_eq!(functions[0].return_type.as_deref(), Some("i64"));
            }
            other => panic!("expected ExternBlock, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_extern_block_ptr_type() {
        let prog = parse("extern \"vulkan\" {\n  fn vkCreateInstance(info: ptr, allocator: ptr, instance: ptr) -> u32\n}\n").unwrap();
        match &prog.statements[0].0 {
            Statement::ExternBlock { library, functions, .. } => {
                assert_eq!(library, "vulkan");
                assert_eq!(functions[0].params.len(), 3);
                for p in &functions[0].params {
                    assert_eq!(p.type_name, "ptr");
                }
                assert_eq!(functions[0].return_type.as_deref(), Some("u32"));
            }
            other => panic!("expected ExternBlock, got {:?}", other),
        }
    }
}
