/// Lightweight JavaScript execution engine for OctoView Browser.
///
/// Wraps the Boa engine to execute inline `<script>` tags with basic DOM-like
/// APIs.  Instead of building a full mutable DOM tree (extremely complex), we
/// use a **mutation log pattern**:
///
/// 1. JS calls `document.getElementById("foo").setInnerHTML("bar")`
/// 2. Our native function captures this as `DomMutation::SetInnerHtml`
/// 3. After all scripts execute, mutations are applied to the OVD document.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, Ordering};

use boa_engine::object::builtins::{JsArray, JsPromise};
use boa_engine::object::ObjectInitializer;
use boa_engine::property::Attribute;
use boa_engine::{js_string, Context, JsResult, JsValue, NativeFunction, Source};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A DOM mutation that JS wants to perform.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields read when applying mutations to OVD
pub enum DomMutation {
    /// Set innerHTML of an element by ID.
    SetInnerHtml { id: String, html: String },
    /// Set textContent of an element by ID.
    SetTextContent { id: String, text: String },
    /// Add a class to an element by ID.
    AddClass { id: String, class: String },
    /// Remove a class from an element by ID.
    RemoveClass { id: String, class: String },
    /// Set a style property on an element by ID.
    SetStyle {
        id: String,
        property: String,
        value: String,
    },
    /// Set display:none on an element by ID.
    Hide { id: String },
    /// Remove display:none on an element by ID.
    Show { id: String },
    /// Write text to the document (`document.write`).
    DocumentWrite { html: String },
    /// Console log message.
    ConsoleLog { message: String },
    /// Create a new element with a tag and generated ID.
    CreateElement { tag: String, id: String },
    /// Append a child element to a parent.
    AppendChild { parent_id: String, child_id: String },
    /// Create a text node with content.
    CreateTextNode { text: String, id: String },
    /// Set an attribute on an element.
    SetAttribute {
        id: String,
        name: String,
        value: String,
    },
}

/// A stored event listener — callback stored as JS source string.
///
/// Since Boa's `JsFunction` is not `Send`/`Copy`, we store the callback as
/// its source representation.  When the event fires, we re-execute the source
/// string in a new JS context.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct StoredListener {
    /// Event key: either `"click"` for document-level or `"#myid:click"` for element-level.
    pub event_key: String,
    /// The JS source of the callback function.
    pub callback_source: String,
}

/// A pending timer created by `setTimeout` or `setInterval`.
#[derive(Debug, Clone)]
pub struct TimerEntry {
    /// Unique timer ID (returned to JS).
    pub id: u32,
    /// The JS source of the callback function.
    pub callback_source: String,
    /// Delay in milliseconds.
    pub delay_ms: u64,
    /// `true` for `setInterval`, `false` for `setTimeout`.
    pub is_interval: bool,
}

/// Result of executing JavaScript.
#[derive(Debug, Default)]
pub struct JsExecutionResult {
    pub mutations: Vec<DomMutation>,
    pub errors: Vec<String>,
    /// Event listeners registered during script execution.
    pub listeners: Vec<StoredListener>,
    /// Pending timers (setTimeout/setInterval) registered during script execution.
    pub timers: Vec<TimerEntry>,
}

// ---------------------------------------------------------------------------
// Thread-local mutation log
//
// `NativeFunction::from_copy_closure` requires `Copy` closures, which rules
// out capturing `Rc<RefCell<..>>`.  The `from_closure` alternative is `unsafe`.
// Using a thread-local side-channel is the simplest *safe* approach -- and
// perfectly fine because script execution is single-threaded.
// ---------------------------------------------------------------------------

thread_local! {
    static MUTATIONS: RefCell<Vec<DomMutation>> = RefCell::new(Vec::new());
    static EVENT_LISTENERS: RefCell<Vec<StoredListener>> = RefCell::new(Vec::new());
    static TIMERS: RefCell<Vec<TimerEntry>> = RefCell::new(Vec::new());
}

/// Global timer ID counter (atomic so it works across threads if needed).
static NEXT_TIMER_ID: AtomicU32 = AtomicU32::new(1);

fn push_mutation(m: DomMutation) {
    MUTATIONS.with(|ms| ms.borrow_mut().push(m));
}

fn take_mutations() -> Vec<DomMutation> {
    MUTATIONS.with(|ms| ms.take())
}

fn push_listener(listener: StoredListener) {
    EVENT_LISTENERS.with(|ls| ls.borrow_mut().push(listener));
}

fn take_listeners() -> Vec<StoredListener> {
    EVENT_LISTENERS.with(|ls| ls.take())
}

fn push_timer(timer: TimerEntry) {
    TIMERS.with(|ts| ts.borrow_mut().push(timer));
}

fn take_timers() -> Vec<TimerEntry> {
    TIMERS.with(|ts| ts.take())
}

fn remove_timer(id: u32) {
    TIMERS.with(|ts| ts.borrow_mut().retain(|t| t.id != id));
}

// ---------------------------------------------------------------------------
// Core entry point
// ---------------------------------------------------------------------------

/// Counter for unique callback variable names.
static NEXT_CALLBACK_ID: AtomicU32 = AtomicU32::new(1);

/// Store a JS callback by assigning it to a unique global variable.
///
/// Boa 0.20 does not support `Function.prototype.toString()` returning source
/// text (it returns `"function () { [native code] }"` for all functions).
/// Instead, we store the callback by assigning it to a uniquely-named global
/// variable (e.g., `__ov_cb_1`) and return `"__ov_cb_1()"` as the source to
/// execute later.
///
/// If the argument is already a string (e.g., `setTimeout("code", ms)`), we
/// store it directly.
fn extract_callback_source(arg: &JsValue, context: &mut Context) -> JsResult<String> {
    if arg.is_string() {
        // String argument: store as-is (e.g., setTimeout("alert('hi')", 1000))
        let s = arg.to_string(context)?.to_std_string_escaped();
        return Ok(s);
    }
    if arg.is_object() {
        // Function argument: assign to a global and store the call expression.
        let cb_id = NEXT_CALLBACK_ID.fetch_add(1, Ordering::Relaxed);
        let var_name = format!("__ov_cb_{cb_id}");
        // Register the callback as a global property so it persists in the context.
        context.register_global_property(
            boa_engine::JsString::from(var_name.as_str()),
            arg.clone(),
            Attribute::all(),
        )?;
        return Ok(format!("{var_name}()"));
    }
    // Fallback: convert to string.
    let s = arg.to_string(context)?.to_std_string_escaped();
    Ok(s)
}

/// Maximum script size we'll attempt to execute (64 KB).
/// YouTube/Google inline scripts can be 500KB+, which freezes Boa for minutes.
const MAX_SCRIPT_SIZE: usize = 64 * 1024;

/// Maximum number of scripts to execute per page.
const MAX_SCRIPTS: usize = 10;

/// Execute a list of JavaScript source strings and return DOM mutations.
pub fn execute_scripts(scripts: &[String]) -> JsExecutionResult {
    // Clear any stale state from a previous call.
    let _ = take_mutations();
    let _ = take_listeners();
    let _ = take_timers();

    let mut context = Context::default();
    let mut errors: Vec<String> = Vec::new();

    // Register browser-like globals.
    if let Err(e) = register_console(&mut context) {
        errors.push(format!("Failed to register console: {e}"));
    }
    if let Err(e) = register_document(&mut context) {
        errors.push(format!("Failed to register document: {e}"));
    }
    if let Err(e) = register_alert(&mut context) {
        errors.push(format!("Failed to register alert: {e}"));
    }
    if let Err(e) = register_window(&mut context) {
        errors.push(format!("Failed to register window: {e}"));
    }

    // Register global setTimeout/setInterval/clearTimeout/clearInterval
    // (also accessible as window.setTimeout etc, but many scripts call them globally)
    register_timer_globals(&mut context, &mut errors);

    // Register global fetch() API (synchronous bridge using reqwest::blocking)
    if let Err(e) = register_fetch(&mut context) {
        errors.push(format!("Failed to register fetch: {e}"));
    }

    // Execute each script (with size and count caps).
    for script in scripts.iter().take(MAX_SCRIPTS) {
        if script.len() > MAX_SCRIPT_SIZE {
            errors.push(format!(
                "Script skipped ({} KB exceeds {} KB limit)",
                script.len() / 1024,
                MAX_SCRIPT_SIZE / 1024
            ));
            continue;
        }
        match context.eval(Source::from_bytes(script.as_bytes())) {
            Ok(_) => {}
            Err(e) => {
                errors.push(format!("{e}"));
            }
        }
    }

    // Execute any setTimeout(fn, 0) callbacks immediately (common pattern for deferred execution).
    let timers = take_timers();
    let mut remaining_timers = Vec::new();
    for timer in timers {
        if timer.delay_ms == 0 && !timer.is_interval {
            // callback_source is either raw code or "__ov_cb_N()" call expression.
            match context.eval(Source::from_bytes(timer.callback_source.as_bytes())) {
                Ok(_) => {}
                Err(e) => errors.push(format!("setTimeout(0) error: {e}")),
            }
        } else {
            remaining_timers.push(timer);
        }
    }

    // Collect all mutations (from scripts + timer callbacks).
    let all_mutations = take_mutations();

    JsExecutionResult {
        mutations: all_mutations,
        errors,
        listeners: take_listeners(),
        timers: remaining_timers,
    }
}

// ---------------------------------------------------------------------------
// Global timer functions (setTimeout, setInterval, clearTimeout, clearInterval)
// ---------------------------------------------------------------------------

fn register_timer_globals(context: &mut Context, errors: &mut Vec<String>) {
    // setTimeout(callback, delay) -> timer_id
    let set_timeout_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let callback_source = args
            .first()
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();
        let delay_ms = args
            .get(1)
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n.max(0.0) as u64)
            .unwrap_or(0);
        let id = NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed);
        push_timer(TimerEntry {
            id,
            callback_source,
            delay_ms,
            is_interval: false,
        });
        Ok(JsValue::from(id))
    });

    // setInterval(callback, delay) -> timer_id
    let set_interval_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let callback_source = args
            .first()
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();
        let delay_ms = args
            .get(1)
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n.max(0.0) as u64)
            .unwrap_or(0);
        let id = NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed);
        push_timer(TimerEntry {
            id,
            callback_source,
            delay_ms,
            is_interval: true,
        });
        Ok(JsValue::from(id))
    });

    // clearTimeout(id)
    let clear_timeout_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let id = args
            .first()
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n as u32)
            .unwrap_or(0);
        remove_timer(id);
        Ok(JsValue::undefined())
    });

    // clearInterval(id) -- same logic as clearTimeout
    let clear_interval_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let id = args
            .first()
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n as u32)
            .unwrap_or(0);
        remove_timer(id);
        Ok(JsValue::undefined())
    });

    // Register as globals (many scripts call setTimeout directly, not window.setTimeout)
    if let Err(e) = context.register_global_property(
        js_string!("setTimeout"),
        set_timeout_fn.to_js_function(context.realm()),
        Attribute::all(),
    ) {
        errors.push(format!("Failed to register setTimeout: {e}"));
    }
    if let Err(e) = context.register_global_property(
        js_string!("setInterval"),
        set_interval_fn.to_js_function(context.realm()),
        Attribute::all(),
    ) {
        errors.push(format!("Failed to register setInterval: {e}"));
    }
    if let Err(e) = context.register_global_property(
        js_string!("clearTimeout"),
        clear_timeout_fn.to_js_function(context.realm()),
        Attribute::all(),
    ) {
        errors.push(format!("Failed to register clearTimeout: {e}"));
    }
    if let Err(e) = context.register_global_property(
        js_string!("clearInterval"),
        clear_interval_fn.to_js_function(context.realm()),
        Attribute::all(),
    ) {
        errors.push(format!("Failed to register clearInterval: {e}"));
    }
}

// ---------------------------------------------------------------------------
// console object
// ---------------------------------------------------------------------------

fn register_console(context: &mut Context) -> JsResult<()> {
    // console.log(...)
    let log_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let msg = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        push_mutation(DomMutation::ConsoleLog { message: msg });
        Ok(JsValue::undefined())
    });

    // console.warn(...)  -- treat same as log for now
    let warn_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let msg = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        push_mutation(DomMutation::ConsoleLog {
            message: format!("[warn] {msg}"),
        });
        Ok(JsValue::undefined())
    });

    let console = ObjectInitializer::new(context)
        .function(log_fn, js_string!("log"), 1)
        .function(warn_fn, js_string!("warn"), 1)
        .build();

    context.register_global_property(js_string!("console"), console, Attribute::all())
}

// ---------------------------------------------------------------------------
// document object
// ---------------------------------------------------------------------------

fn register_document(context: &mut Context) -> JsResult<()> {
    // document.getElementById(id) -> element proxy
    let get_element_fn =
        NativeFunction::from_copy_closure(|_this, args, ctx| {
            let id = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();

            if id.is_empty() {
                return Ok(JsValue::null());
            }

            // Build a lightweight element proxy.
            Ok(JsValue::from(build_element_proxy(ctx, &id)))
        });

    // document.write(html)
    let write_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let html = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        push_mutation(DomMutation::DocumentWrite { html });
        Ok(JsValue::undefined())
    });

    // document.querySelector(sel) -> null (stub)
    let query_fn = NativeFunction::from_copy_closure(|_this, _args, _ctx| {
        Ok(JsValue::null())
    });

    // document.querySelectorAll(sel) -> empty array (stub, prevents crashes)
    let query_all_fn = NativeFunction::from_copy_closure(|_this, _args, ctx| {
        let arr = JsArray::new(ctx);
        Ok(JsValue::from(arr))
    });

    // document.addEventListener(event, callback)
    // "DOMContentLoaded" callbacks execute immediately (DOM is already loaded
    // by the time our scripts run).  Other events are stored for later.
    let doc_add_event_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let event = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let callback_source = args
            .get(1)
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();

        if event == "DOMContentLoaded" {
            // Execute immediately -- DOM is already loaded.
            // callback_source is either raw code or "__ov_cb_N()" call expression.
            match ctx.eval(Source::from_bytes(callback_source.as_bytes())) {
                Ok(_) => {}
                Err(e) => {
                    push_mutation(DomMutation::ConsoleLog {
                        message: format!("[DOMContentLoaded error] {e}"),
                    });
                }
            }
        } else {
            push_listener(StoredListener {
                event_key: event,
                callback_source,
            });
        }
        Ok(JsValue::undefined())
    });

    // document.createElement(tag) -> element proxy with generated ID
    let create_element_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let tag = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let id = next_generated_id();
        push_mutation(DomMutation::CreateElement {
            tag: tag.clone(),
            id: id.clone(),
        });
        Ok(JsValue::from(build_element_proxy(ctx, &id)))
    });

    // document.createTextNode(text) -> element proxy with generated ID
    let create_text_node_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let text = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let id = next_generated_id();
        push_mutation(DomMutation::CreateTextNode {
            text,
            id: id.clone(),
        });
        Ok(JsValue::from(build_element_proxy(ctx, &id)))
    });

    // document.body -> element proxy with id "body"
    let body_proxy = build_element_proxy(context, "body");

    // document.head -> element proxy with id "head"
    let head_proxy = build_element_proxy(context, "head");

    let document = ObjectInitializer::new(context)
        .function(get_element_fn, js_string!("getElementById"), 1)
        .function(write_fn, js_string!("write"), 1)
        .function(query_fn, js_string!("querySelector"), 1)
        .function(query_all_fn, js_string!("querySelectorAll"), 1)
        .function(doc_add_event_fn, js_string!("addEventListener"), 2)
        .function(create_element_fn, js_string!("createElement"), 1)
        .function(create_text_node_fn, js_string!("createTextNode"), 1)
        .property(js_string!("body"), body_proxy, Attribute::all())
        .property(js_string!("head"), head_proxy, Attribute::all())
        .build();

    context.register_global_property(js_string!("document"), document, Attribute::all())
}

// ---------------------------------------------------------------------------
// alert() global
// ---------------------------------------------------------------------------

fn register_alert(context: &mut Context) -> JsResult<()> {
    let alert_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let msg = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        push_mutation(DomMutation::ConsoleLog {
            message: format!("[alert] {msg}"),
        });
        Ok(JsValue::undefined())
    });

    context.register_global_property(js_string!("alert"), alert_fn.to_js_function(context.realm()), Attribute::all())
}

// ---------------------------------------------------------------------------
// fetch() API  (synchronous bridge)
//
// Implements `window.fetch(url)` using `reqwest::blocking` under the hood.
// Returns a pre-resolved Promise with a Response-like object containing
// `.text()`, `.json()`, `.ok`, and `.status`.
//
// Limitations:
// - Synchronous: blocks the JS thread during the HTTP request
// - Body capped at 256 KB
// - 5-second timeout
// ---------------------------------------------------------------------------

/// Maximum fetch response body size (256 KB).
const MAX_FETCH_BODY: usize = 256 * 1024;

/// Fetch timeout in seconds.
const FETCH_TIMEOUT_SECS: u64 = 5;

fn register_fetch(context: &mut Context) -> JsResult<()> {
    let fetch_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let url = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();

        if url.is_empty() {
            // Return a rejected-like response.
            let response = build_fetch_error_response(ctx, "fetch: empty URL");
            let promise = JsPromise::resolve(JsValue::from(response), ctx);
            return Ok(JsValue::from(promise));
        }

        // Perform synchronous HTTP request.
        match do_fetch_sync(&url) {
            Ok((status, body)) => {
                let response = build_fetch_response(ctx, status, &body);
                let promise = JsPromise::resolve(JsValue::from(response), ctx);
                Ok(JsValue::from(promise))
            }
            Err(err_msg) => {
                let response = build_fetch_error_response(ctx, &err_msg);
                let promise = JsPromise::resolve(JsValue::from(response), ctx);
                Ok(JsValue::from(promise))
            }
        }
    });

    context.register_global_property(
        js_string!("fetch"),
        fetch_fn.to_js_function(context.realm()),
        Attribute::all(),
    )
}

/// Perform a synchronous HTTP GET request.  Returns (status_code, body_string).
fn do_fetch_sync(url: &str) -> Result<(u16, String), String> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("OctoView/0.1")
        .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("fetch client error: {e}"))?;

    let response = client
        .get(url)
        .send()
        .map_err(|e| format!("fetch failed: {e}"))?;

    let status = response.status().as_u16();

    // Read body with size cap.
    let bytes = response
        .bytes()
        .map_err(|e| format!("fetch read body: {e}"))?;

    if bytes.len() > MAX_FETCH_BODY {
        return Err(format!(
            "fetch body too large: {} KB (max {} KB)",
            bytes.len() / 1024,
            MAX_FETCH_BODY / 1024
        ));
    }

    let body = String::from_utf8_lossy(&bytes).to_string();
    Ok((status, body))
}

/// Build a Response-like JS object with .ok, .status, .text(), .json().
fn build_fetch_response(context: &mut Context, status: u16, body: &str) -> boa_engine::JsObject {
    let ok = (200..300).contains(&status);
    let body_js = JsValue::from(js_string!(body));

    // .text() -> Promise resolving to body string
    // We store the body on the response object and read it from `this`.
    let text_fn = NativeFunction::from_copy_closure(|this, _args, ctx| {
        let body_val = match this {
            JsValue::Object(obj) => obj.get(js_string!("_body"), ctx)?,
            _ => JsValue::from(js_string!("")),
        };
        let promise = JsPromise::resolve(body_val, ctx);
        Ok(JsValue::from(promise))
    });

    // .json() -> Promise resolving to parsed JSON
    let json_fn = NativeFunction::from_copy_closure(|this, _args, ctx| {
        let body_str = match this {
            JsValue::Object(obj) => {
                let val = obj.get(js_string!("_body"), ctx)?;
                val.to_string(ctx)?.to_std_string_escaped()
            }
            _ => String::new(),
        };
        // Use ctx.eval to parse JSON (leverages Boa's built-in JSON.parse).
        let parse_code = format!("JSON.parse({})", json_quote(&body_str));
        match ctx.eval(Source::from_bytes(parse_code.as_bytes())) {
            Ok(parsed) => {
                let promise = JsPromise::resolve(parsed, ctx);
                Ok(JsValue::from(promise))
            }
            Err(e) => {
                // Return a resolved promise with the error message.
                let err_msg = JsValue::from(js_string!(format!("JSON parse error: {e}").as_str()));
                let promise = JsPromise::resolve(err_msg, ctx);
                Ok(JsValue::from(promise))
            }
        }
    });

    ObjectInitializer::new(context)
        .property(js_string!("ok"), JsValue::from(ok), Attribute::all())
        .property(js_string!("status"), JsValue::from(status as i32), Attribute::all())
        .property(js_string!("_body"), body_js, Attribute::CONFIGURABLE)
        .function(text_fn, js_string!("text"), 0)
        .function(json_fn, js_string!("json"), 0)
        .build()
}

/// Build an error Response-like JS object (status 0, ok false).
fn build_fetch_error_response(context: &mut Context, error: &str) -> boa_engine::JsObject {
    let body_js = JsValue::from(js_string!(""));
    let err_js = JsValue::from(js_string!(error));

    let text_fn = NativeFunction::from_copy_closure(|_this, _args, ctx| {
        let promise = JsPromise::resolve(JsValue::from(js_string!("")), ctx);
        Ok(JsValue::from(promise))
    });

    let json_fn = NativeFunction::from_copy_closure(|_this, _args, ctx| {
        let promise = JsPromise::resolve(JsValue::null(), ctx);
        Ok(JsValue::from(promise))
    });

    ObjectInitializer::new(context)
        .property(js_string!("ok"), JsValue::from(false), Attribute::all())
        .property(js_string!("status"), JsValue::from(0), Attribute::all())
        .property(js_string!("_body"), body_js, Attribute::CONFIGURABLE)
        .property(js_string!("error"), err_js, Attribute::all())
        .function(text_fn, js_string!("text"), 0)
        .function(json_fn, js_string!("json"), 0)
        .build()
}

/// Escape a string for use as a JSON string literal in JS source.
fn json_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c < ' ' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ---------------------------------------------------------------------------
// window object
//
// Many scripts check `window.location`, `window.addEventListener`, etc.
// We provide a minimal stub so scripts don't immediately throw
// "window is not defined".
// ---------------------------------------------------------------------------

fn register_window(context: &mut Context) -> JsResult<()> {
    // No-op function for non-essential stubs.
    let noop_fn = NativeFunction::from_copy_closure(|_this, _args, _ctx| {
        Ok(JsValue::undefined())
    });

    // window.addEventListener -- stores listeners (same logic as document.addEventListener)
    let win_add_event_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let event = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let callback_source = args
            .get(1)
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();

        if event == "DOMContentLoaded" || event == "load" {
            // Execute immediately.
            match ctx.eval(Source::from_bytes(callback_source.as_bytes())) {
                Ok(_) => {}
                Err(e) => {
                    push_mutation(DomMutation::ConsoleLog {
                        message: format!("[window.{event} error] {e}"),
                    });
                }
            }
        } else {
            push_listener(StoredListener {
                event_key: format!("window:{event}"),
                callback_source,
            });
        }
        Ok(JsValue::undefined())
    });

    // setTimeout/setInterval/clearTimeout/clearInterval -- use the same logic
    // as the global functions (they delegate to the thread-local TIMERS).
    let win_set_timeout = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let callback_source = args
            .first()
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();
        let delay_ms = args
            .get(1)
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n.max(0.0) as u64)
            .unwrap_or(0);
        let id = NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed);
        push_timer(TimerEntry {
            id,
            callback_source,
            delay_ms,
            is_interval: false,
        });
        Ok(JsValue::from(id))
    });

    let win_set_interval = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let callback_source = args
            .first()
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();
        let delay_ms = args
            .get(1)
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n.max(0.0) as u64)
            .unwrap_or(0);
        let id = NEXT_TIMER_ID.fetch_add(1, Ordering::Relaxed);
        push_timer(TimerEntry {
            id,
            callback_source,
            delay_ms,
            is_interval: true,
        });
        Ok(JsValue::from(id))
    });

    let win_clear_timeout = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let id = args
            .first()
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n as u32)
            .unwrap_or(0);
        remove_timer(id);
        Ok(JsValue::undefined())
    });

    let win_clear_interval = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let id = args
            .first()
            .map(|v| v.to_number(ctx))
            .transpose()?
            .map(|n| n as u32)
            .unwrap_or(0);
        remove_timer(id);
        Ok(JsValue::undefined())
    });

    // location stub
    let location = ObjectInitializer::new(context)
        .property(js_string!("href"), js_string!(""), Attribute::all())
        .property(js_string!("hostname"), js_string!(""), Attribute::all())
        .property(js_string!("pathname"), js_string!("/"), Attribute::all())
        .property(js_string!("protocol"), js_string!("https:"), Attribute::all())
        .property(js_string!("search"), js_string!(""), Attribute::all())
        .property(js_string!("hash"), js_string!(""), Attribute::all())
        .build();

    // navigator stub
    let navigator = ObjectInitializer::new(context)
        .property(
            js_string!("userAgent"),
            js_string!("OctoView/0.1"),
            Attribute::all(),
        )
        .property(js_string!("language"), js_string!("en-US"), Attribute::all())
        .build();

    // window.fetch -- delegates to the global fetch() (same implementation)
    let win_fetch_fn = NativeFunction::from_copy_closure(|_this, args, ctx| {
        let url = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();

        if url.is_empty() {
            let response = build_fetch_error_response(ctx, "fetch: empty URL");
            let promise = JsPromise::resolve(JsValue::from(response), ctx);
            return Ok(JsValue::from(promise));
        }

        match do_fetch_sync(&url) {
            Ok((status, body)) => {
                let response = build_fetch_response(ctx, status, &body);
                let promise = JsPromise::resolve(JsValue::from(response), ctx);
                Ok(JsValue::from(promise))
            }
            Err(err_msg) => {
                let response = build_fetch_error_response(ctx, &err_msg);
                let promise = JsPromise::resolve(JsValue::from(response), ctx);
                Ok(JsValue::from(promise))
            }
        }
    });

    let window = ObjectInitializer::new(context)
        .function(win_add_event_fn, js_string!("addEventListener"), 3)
        .function(noop_fn.clone(), js_string!("removeEventListener"), 3)
        .function(win_set_timeout, js_string!("setTimeout"), 2)
        .function(win_set_interval, js_string!("setInterval"), 2)
        .function(win_clear_timeout, js_string!("clearTimeout"), 1)
        .function(win_clear_interval, js_string!("clearInterval"), 1)
        .function(noop_fn.clone(), js_string!("requestAnimationFrame"), 1)
        .function(noop_fn, js_string!("getComputedStyle"), 1)
        .function(win_fetch_fn, js_string!("fetch"), 2)
        .property(js_string!("location"), location, Attribute::all())
        .property(js_string!("navigator"), navigator, Attribute::all())
        .property(js_string!("innerWidth"), JsValue::from(1280), Attribute::all())
        .property(js_string!("innerHeight"), JsValue::from(800), Attribute::all())
        .build();

    context.register_global_property(js_string!("window"), window, Attribute::all())?;

    // Also register navigator at global scope (scripts access both window.navigator and navigator)
    let navigator2 = ObjectInitializer::new(context)
        .property(
            js_string!("userAgent"),
            js_string!("OctoView/0.1"),
            Attribute::all(),
        )
        .property(js_string!("language"), js_string!("en-US"), Attribute::all())
        .build();
    context.register_global_property(js_string!("navigator"), navigator2, Attribute::all())
}

// ---------------------------------------------------------------------------
// Generated element ID counter
// ---------------------------------------------------------------------------

/// Counter for auto-generated element IDs (createElement, createTextNode).
static NEXT_ELEMENT_ID: AtomicU32 = AtomicU32::new(1);

fn next_generated_id() -> String {
    let n = NEXT_ELEMENT_ID.fetch_add(1, Ordering::Relaxed);
    format!("__ov_el_{n}")
}

// ---------------------------------------------------------------------------
// Element proxy builder
//
// Because `NativeFunction::from_copy_closure` requires `Copy` closures, we
// cannot capture the element `id` directly.  Instead each method reads the
// `id` property that is stored on the proxy object itself (available via
// `this`).
// ---------------------------------------------------------------------------

fn build_element_proxy(context: &mut Context, id: &str) -> boa_engine::JsObject {
    // --- setInnerHTML(html) ---
    let set_html_fn =
        NativeFunction::from_copy_closure(|this, args, ctx| {
            let id = read_id_from_this(this, ctx)?;
            let html = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            push_mutation(DomMutation::SetInnerHtml { id, html });
            Ok(JsValue::undefined())
        });

    // --- setTextContent(text) ---
    let set_text_fn =
        NativeFunction::from_copy_closure(|this, args, ctx| {
            let id = read_id_from_this(this, ctx)?;
            let text = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            push_mutation(DomMutation::SetTextContent { id, text });
            Ok(JsValue::undefined())
        });

    // --- addClass(cls) ---
    let add_class_fn =
        NativeFunction::from_copy_closure(|this, args, ctx| {
            let id = read_id_from_this(this, ctx)?;
            let class = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            push_mutation(DomMutation::AddClass { id, class });
            Ok(JsValue::undefined())
        });

    // --- removeClass(cls) ---
    let remove_class_fn =
        NativeFunction::from_copy_closure(|this, args, ctx| {
            let id = read_id_from_this(this, ctx)?;
            let class = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            push_mutation(DomMutation::RemoveClass { id, class });
            Ok(JsValue::undefined())
        });

    // --- setStyle(property, value) ---
    let set_style_fn =
        NativeFunction::from_copy_closure(|this, args, ctx| {
            let id = read_id_from_this(this, ctx)?;
            let property = args
                .first()
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            let value = args
                .get(1)
                .map(|v| v.to_string(ctx))
                .transpose()?
                .map(|s| s.to_std_string_escaped())
                .unwrap_or_default();
            push_mutation(DomMutation::SetStyle {
                id,
                property,
                value,
            });
            Ok(JsValue::undefined())
        });

    // --- hide() ---
    let hide_fn = NativeFunction::from_copy_closure(|this, _args, ctx| {
        let id = read_id_from_this(this, ctx)?;
        push_mutation(DomMutation::Hide { id });
        Ok(JsValue::undefined())
    });

    // --- show() ---
    let show_fn = NativeFunction::from_copy_closure(|this, _args, ctx| {
        let id = read_id_from_this(this, ctx)?;
        push_mutation(DomMutation::Show { id });
        Ok(JsValue::undefined())
    });

    // --- addEventListener(event, callback) ---
    let add_event_fn = NativeFunction::from_copy_closure(|this, args, ctx| {
        let id = read_id_from_this(this, ctx)?;
        let event = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let callback_source = args
            .get(1)
            .map(|v| extract_callback_source(v, ctx))
            .transpose()?
            .unwrap_or_default();
        // Store with element ID prefix: "#myid:click"
        let event_key = format!("#{id}:{event}");
        push_listener(StoredListener {
            event_key,
            callback_source,
        });
        Ok(JsValue::undefined())
    });

    // --- appendChild(child) ---
    let append_child_fn = NativeFunction::from_copy_closure(|this, args, ctx| {
        let parent_id = read_id_from_this(this, ctx)?;
        let child_id = match args.first() {
            Some(JsValue::Object(obj)) => {
                let val = obj.get(js_string!("id"), ctx)?;
                val.to_string(ctx)?.to_std_string_escaped()
            }
            _ => String::new(),
        };
        if !child_id.is_empty() {
            push_mutation(DomMutation::AppendChild {
                parent_id,
                child_id,
            });
        }
        // Return the child argument (standard DOM behavior).
        Ok(args.first().cloned().unwrap_or(JsValue::undefined()))
    });

    // --- setAttribute(name, value) ---
    let set_attr_fn = NativeFunction::from_copy_closure(|this, args, ctx| {
        let id = read_id_from_this(this, ctx)?;
        let name = args
            .first()
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        let value = args
            .get(1)
            .map(|v| v.to_string(ctx))
            .transpose()?
            .map(|s| s.to_std_string_escaped())
            .unwrap_or_default();
        push_mutation(DomMutation::SetAttribute { id, name, value });
        Ok(JsValue::undefined())
    });

    // --- getAttribute(name) -> "" (stub) ---
    let get_attr_fn = NativeFunction::from_copy_closure(|_this, _args, _ctx| {
        Ok(JsValue::from(js_string!("")))
    });

    let id_js = JsValue::from(boa_engine::JsString::from(id));

    ObjectInitializer::new(context)
        .property(js_string!("id"), id_js, Attribute::READONLY)
        .function(set_html_fn, js_string!("setInnerHTML"), 1)
        .function(set_text_fn, js_string!("setTextContent"), 1)
        .function(add_class_fn, js_string!("addClass"), 1)
        .function(remove_class_fn, js_string!("removeClass"), 1)
        .function(set_style_fn, js_string!("setStyle"), 2)
        .function(hide_fn, js_string!("hide"), 0)
        .function(show_fn, js_string!("show"), 0)
        .function(add_event_fn, js_string!("addEventListener"), 2)
        .function(append_child_fn, js_string!("appendChild"), 1)
        .function(set_attr_fn, js_string!("setAttribute"), 2)
        .function(get_attr_fn, js_string!("getAttribute"), 1)
        .build()
}

/// Helper: read the `id` property from the `this` binding of an element proxy method.
fn read_id_from_this(this: &JsValue, context: &mut Context) -> JsResult<String> {
    match this {
        JsValue::Object(obj) => {
            let val = obj.get(js_string!("id"), context)?;
            Ok(val
                .to_string(context)?
                .to_std_string_escaped())
        }
        _ => Ok(String::new()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_console_log() {
        let result = execute_scripts(&["console.log('hello')".to_string()]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.mutations.len(), 1);
        match &result.mutations[0] {
            DomMutation::ConsoleLog { message } => assert_eq!(message, "hello"),
            other => panic!("Expected ConsoleLog, got {:?}", other),
        }
    }

    #[test]
    fn test_document_write() {
        let result = execute_scripts(&["document.write('<p>hi</p>')".to_string()]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.mutations.len(), 1);
        match &result.mutations[0] {
            DomMutation::DocumentWrite { html } => assert_eq!(html, "<p>hi</p>"),
            other => panic!("Expected DocumentWrite, got {:?}", other),
        }
    }

    #[test]
    fn test_get_element_by_id() {
        let result = execute_scripts(&[
            "var el = document.getElementById('foo'); console.log(el !== null ? 'found' : 'null');"
                .to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "found"
        )));
    }

    #[test]
    fn test_set_inner_html() {
        let result = execute_scripts(&[
            "var el = document.getElementById('foo'); el.setInnerHTML('<b>bold</b>');".to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::SetInnerHtml { id, html } if id == "foo" && html == "<b>bold</b>"
        )));
    }

    #[test]
    fn test_set_text_content() {
        let result = execute_scripts(&[
            "var el = document.getElementById('bar'); el.setTextContent('plain text');".to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::SetTextContent { id, text } if id == "bar" && text == "plain text"
        )));
    }

    #[test]
    fn test_multiple_scripts() {
        let scripts = vec![
            "console.log('first')".to_string(),
            "console.log('second')".to_string(),
        ];
        let result = execute_scripts(&scripts);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.mutations.len(), 2);
        match &result.mutations[0] {
            DomMutation::ConsoleLog { message } => assert_eq!(message, "first"),
            other => panic!("Expected ConsoleLog, got {:?}", other),
        }
        match &result.mutations[1] {
            DomMutation::ConsoleLog { message } => assert_eq!(message, "second"),
            other => panic!("Expected ConsoleLog, got {:?}", other),
        }
    }

    #[test]
    fn test_js_error_captured() {
        let result = execute_scripts(&["this is not valid javascript {{{}}}".to_string()]);
        assert!(
            !result.errors.is_empty(),
            "Expected at least one error for invalid JS"
        );
    }

    #[test]
    fn test_arithmetic() {
        let result = execute_scripts(&["var x = 1 + 2; console.log(x);".to_string()]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "3"
        )));
    }

    #[test]
    fn test_get_element_empty_returns_null() {
        let result = execute_scripts(&[
            "var el = document.getElementById(''); console.log(el === null ? 'isnull' : 'notnull');"
                .to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "isnull"
        )));
    }

    #[test]
    fn test_alert_captured() {
        let result = execute_scripts(&["alert('hello world')".to_string()]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message.contains("hello world")
        )));
    }

    // --- Feature 1: Event Listener tests ---

    #[test]
    fn test_dom_content_loaded() {
        // DOMContentLoaded callback should execute immediately.
        let result = execute_scripts(&[
            r#"document.addEventListener("DOMContentLoaded", function() { console.log("loaded"); });"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // The callback should have executed, producing a ConsoleLog mutation.
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "loaded"
        )), "DOMContentLoaded callback should execute immediately, mutations: {:?}", result.mutations);
    }

    #[test]
    fn test_add_event_listener_stores() {
        // A non-DOMContentLoaded listener should be stored and returned.
        let result = execute_scripts(&[
            r#"document.addEventListener("click", function() { console.log("clicked"); });"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.listeners.len(), 1, "expected 1 listener, got {:?}", result.listeners);
        assert_eq!(result.listeners[0].event_key, "click");
        // Callback is stored as a global variable call (Boa 0.20 limitation:
        // Function.prototype.toString() doesn't return source text).
        assert!(result.listeners[0].callback_source.starts_with("__ov_cb_"),
            "callback_source should be a stored callback reference: {}",
            result.listeners[0].callback_source);
    }

    #[test]
    fn test_element_add_event_listener() {
        // Element-level addEventListener should store with "#id:event" key.
        let result = execute_scripts(&[
            r#"var el = document.getElementById("btn1"); el.addEventListener("click", function() { console.log("btn clicked"); });"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.listeners.len(), 1, "expected 1 listener, got {:?}", result.listeners);
        assert_eq!(result.listeners[0].event_key, "#btn1:click");
    }

    // --- Feature 2: Timer tests ---

    #[test]
    fn test_set_timeout_returns_id() {
        let result = execute_scripts(&[
            r#"var id = setTimeout(function() {}, 1000); console.log(id);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // The returned ID should be a positive number.
        let log_msg = result.mutations.iter().find_map(|m| match m {
            DomMutation::ConsoleLog { message } => Some(message.clone()),
            _ => None,
        });
        assert!(log_msg.is_some(), "expected a console.log with timer id");
        let id_val: u32 = log_msg.unwrap().parse().expect("timer id should be a number");
        assert!(id_val > 0, "timer id should be positive");
    }

    #[test]
    fn test_set_timeout_zero_executes() {
        // setTimeout(fn, 0) should execute immediately after scripts.
        let result = execute_scripts(&[
            r#"setTimeout(function() { console.log("immediate"); }, 0);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "immediate"
        )), "setTimeout(fn, 0) should execute immediately, mutations: {:?}", result.mutations);
        // The timer should NOT be in the remaining timers list.
        assert!(result.timers.is_empty(), "setTimeout(0) should not remain in timers: {:?}", result.timers);
    }

    #[test]
    fn test_clear_timeout() {
        // clearTimeout should remove a pending timer.
        let result = execute_scripts(&[
            r#"var id = setTimeout(function() { console.log("should not run"); }, 5000); clearTimeout(id);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // The timer was cleared, so it should not appear in remaining timers.
        assert!(result.timers.is_empty(), "cleared timer should not remain: {:?}", result.timers);
    }

    #[test]
    fn test_set_interval_stored() {
        // setInterval should create a timer with is_interval=true.
        let result = execute_scripts(&[
            r#"setInterval(function() { console.log("tick"); }, 1000);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert_eq!(result.timers.len(), 1, "expected 1 timer, got {:?}", result.timers);
        assert!(result.timers[0].is_interval, "should be an interval timer");
        assert_eq!(result.timers[0].delay_ms, 1000);
    }

    // --- Feature 3: Expanded DOM API tests ---

    #[test]
    fn test_create_element() {
        let result = execute_scripts(&[
            r#"var div = document.createElement("div"); console.log(div.id);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // Should have a CreateElement mutation.
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::CreateElement { tag, id } if tag == "div" && id.starts_with("__ov_el_")
        )), "expected CreateElement mutation, got: {:?}", result.mutations);
    }

    #[test]
    fn test_append_child() {
        let result = execute_scripts(&[
            r#"
            var parent = document.getElementById("container");
            var child = document.createElement("span");
            parent.appendChild(child);
            "#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::AppendChild { parent_id, child_id }
                if parent_id == "container" && child_id.starts_with("__ov_el_")
        )), "expected AppendChild mutation, got: {:?}", result.mutations);
    }

    #[test]
    fn test_set_attribute() {
        let result = execute_scripts(&[
            r#"var el = document.getElementById("img1"); el.setAttribute("src", "photo.jpg");"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::SetAttribute { id, name, value }
                if id == "img1" && name == "src" && value == "photo.jpg"
        )), "expected SetAttribute mutation, got: {:?}", result.mutations);
    }

    #[test]
    fn test_document_body() {
        let result = execute_scripts(&[
            r#"console.log(document.body.id);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "body"
        )), "document.body.id should be 'body', mutations: {:?}", result.mutations);
    }

    #[test]
    fn test_query_selector_all_returns_array() {
        let result = execute_scripts(&[
            r#"var items = document.querySelectorAll(".item"); console.log(items.length);"#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // Should return an empty array with length 0.
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "0"
        )), "querySelectorAll should return empty array, mutations: {:?}", result.mutations);
    }

    // --- Feature 4: fetch() API tests ---

    #[test]
    fn test_fetch_returns_promise() {
        // fetch() should return a Promise-like object.
        // We test the structure by checking that the result has .ok and .status.
        // Since fetch("") returns an error response with ok=false, status=0.
        let result = execute_scripts(&[
            r#"
            var r = fetch("");
            // fetch returns a Promise; in Boa synchronous context, .then() needs run_jobs.
            // Test that fetch exists and doesn't throw.
            console.log(typeof r);
            "#.to_string(),
        ]);
        assert!(result.errors.is_empty(), "errors: {:?}", result.errors);
        // fetch returns an object (Promise).
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "object"
        )), "fetch should return an object (Promise), mutations: {:?}", result.mutations);
    }

    #[test]
    fn test_fetch_error_handling() {
        // fetch with a completely invalid URL should return an error response
        // without crashing. The promise resolves to a response with ok=false.
        let result = execute_scripts(&[
            r#"
            var r = fetch("not-a-url://invalid");
            console.log("no crash");
            "#.to_string(),
        ]);
        // Should not crash, but may have errors from the fetch itself.
        assert!(result.mutations.iter().any(|m| matches!(
            m,
            DomMutation::ConsoleLog { message } if message == "no crash"
        )), "fetch with invalid URL should not crash, mutations: {:?}", result.mutations);
    }
}
