//! `octoflow new` — project scaffolding with embedded templates.
//!
//! Templates are stored as string constants in the binary, just like SPIR-V kernels.
//! Each template generates a main.flow file with a working example.

use crate::CliError;

/// Available template names and descriptions.
pub const TEMPLATES: &[(&str, &str)] = &[
    ("dashboard", "CSV data -> chart -> PNG"),
    ("script",    "File processing + HTTP calls"),
    ("scraper",   "OctoView web scraping + JSON output"),
    ("api",       "HTTP server with JSON endpoints"),
    ("game",      "Window + game loop + keyboard input"),
    ("ai",        "GGUF model loading + inference"),
    ("blank",     "Empty project"),
];

/// Get the main.flow content for a template.
pub fn get_template(name: &str) -> Result<&'static str, CliError> {
    match name {
        "dashboard" => Ok(TEMPLATE_DASHBOARD),
        "script"    => Ok(TEMPLATE_SCRIPT),
        "scraper"   => Ok(TEMPLATE_SCRAPER),
        "api"       => Ok(TEMPLATE_API),
        "game"      => Ok(TEMPLATE_GAME),
        "ai"        => Ok(TEMPLATE_AI),
        "blank"     => Ok(TEMPLATE_BLANK),
        _ => Err(CliError::Compile(format!(
            "unknown template: '{}'. Available: {}",
            name,
            TEMPLATES.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ")
        ))),
    }
}

/// Get the README content for a template.
pub fn get_readme(name: &str, project_name: &str) -> String {
    format!(
        "# {project}\n\
         \n\
         Created with `octoflow new {template} {project}`.\n\
         \n\
         ## Run\n\
         \n\
         ```\n\
         octoflow run main.flow --allow-read --allow-write --allow-net\n\
         ```\n\
         \n\
         ## Edit\n\
         \n\
         Open `main.flow` in any editor, or ask your AI to modify it.\n\
         The OctoFlow Language Guide: https://github.com/octoflow-lang/octoflow/blob/main/docs/language-guide.md\n",
        project = project_name,
        template = name,
    )
}

// ── Template Contents ─────────────────────────────────────────────────

const TEMPLATE_DASHBOARD: &str = r#"// Dashboard — read CSV data and display summary stats
//
// Run: octoflow run main.flow --allow-read
// Edit the data file path and column names to match your data.

use csv

// Load data
let data = read_csv("data.csv")
let headers = csv_headers(data)
print("Columns: {headers}")

// Get a numeric column (change "value" to your column name)
let values = csv_column(data, "value")
let n = len(values)
print("Rows: {n}")

// Basic stats
let mut total = 0.0
let mut min_val = values[0]
let mut max_val = values[0]
for v in values
    total = total + v
    if v < min_val
        min_val = v
    end
    if v > max_val
        max_val = v
    end
end
let avg = total / n

print("Min: {min_val}")
print("Max: {max_val}")
print("Average: {avg}")
print("Total: {total}")
"#;

const TEMPLATE_SCRIPT: &str = r#"// Script — file processing and data transformation
//
// Run: octoflow run main.flow --allow-read --allow-write

// Read input
let lines = read_lines("input.txt")
print("Read {len(lines)} lines")

// Process each line
let mut results = []
for line in lines
    let trimmed = trim(line)
    if len(trimmed) > 0.0
        push(results, to_upper(trimmed))
    end
end

// Write output
let output = join(results, "\n")
write_file("output.txt", output)
print("Wrote {len(results)} lines to output.txt")
"#;

const TEMPLATE_SCRAPER: &str = r#"// Scraper — fetch a web page and extract data
//
// Run: octoflow run main.flow --allow-net --allow-write

// Fetch the page
let url = "https://example.com"
print("Fetching {url}...")
let html = http_get(url)
print("Got {len(html)} bytes")

// Extract title (simple string parsing)
let title_start = index_of(html, "<title>")
if title_start >= 0.0
    let title_end = index_of(html, "</title>")
    let title = substring(html, title_start + 7, title_end)
    print("Title: {title}")
end

// Save raw HTML
write_file("page.html", html)
print("Saved to page.html")
"#;

const TEMPLATE_API: &str = r#"// API Server — HTTP server with JSON endpoints
//
// Run: octoflow run main.flow --allow-net
// Then visit http://localhost:8080

let port = 8080
print("Starting server on port {port}...")
print("Visit http://localhost:{port}")

let server = http_listen(port)

let mut running = 1.0
while running == 1.0
    let req = http_accept(server)
    let path = map_get(req, "path")
    let method = map_get(req, "method")
    print("{method} {path}")

    if path == "/"
        http_respond_html(req, "<h1>OctoFlow API</h1><p>Try /api/hello</p>")
    elif path == "/api/hello"
        let mut response = map()
        response["message"] = "Hello from OctoFlow!"
        response["version"] = "1.0"
        let json = json_stringify(response)
        http_respond(req, 200, json)
    else
        http_respond(req, 404, "Not Found")
    end
end
"#;

const TEMPLATE_GAME: &str = r#"// Game — window with game loop and keyboard input
//
// Run: octoflow run main.flow --allow-ffi

let width = 640
let height = 480
let win = window_open("OctoFlow Game", width, height)

// Game state
let mut player_x = 320.0
let mut player_y = 240.0
let mut speed = 5.0
let mut running = 1.0

print("Arrow keys to move. Close window to exit.")

while running == 1.0
    // Poll events
    let events = window_poll(win)
    if map_has(events, "closed")
        running = 0.0
    end

    // Handle keyboard
    if map_has(events, "key_left")
        player_x = player_x - speed
    end
    if map_has(events, "key_right")
        player_x = player_x + speed
    end
    if map_has(events, "key_up")
        player_y = player_y - speed
    end
    if map_has(events, "key_down")
        player_y = player_y + speed
    end

    // Keep in bounds
    if player_x < 0.0
        player_x = 0.0
    end
    if player_x >= width
        player_x = width - 1.0
    end
    if player_y < 0.0
        player_y = 0.0
    end
    if player_y >= height
        player_y = height - 1.0
    end

    // Render: clear + draw player as a white square
    let mut pixels = gpu_fill(0.0, width * height * 4)
    let px = int(player_x)
    let py = int(player_y)
    let mut dy = 0.0
    while dy < 10
        let mut dx = 0.0
        while dx < 10
            let x = px + int(dx)
            let y = py + int(dy)
            if x >= 0.0
                if x < width
                    if y >= 0.0
                        if y < height
                            let idx = (y * width + x) * 4
                            pixels[int(idx)] = 255.0
                            pixels[int(idx) + 1] = 255.0
                            pixels[int(idx) + 2] = 255.0
                            pixels[int(idx) + 3] = 255.0
                        end
                    end
                end
            end
            dx = dx + 1.0
        end
        dy = dy + 1.0
    end
    window_draw(win, pixels)

    sleep(16)
end

window_close(win)
print("Game over!")
"#;

const TEMPLATE_AI: &str = r#"// AI — load a GGUF model and run inference
//
// Run: octoflow run main.flow --allow-read --allow-ffi
//
// You need a GGUF model file. Recommended:
//   Qwen 2.5 1.5B Q4_K (~1 GB)
//   https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF

use "gguf"
use "generate"

let model_path = "models/qwen2.5-1.5b-q4_k.gguf"
let prompt = "What is the capital of France?"

print("Loading model: {model_path}")
let result = run_generate(model_path, prompt)
"#;

const TEMPLATE_BLANK: &str = r#"// OctoFlow project
//
// Run: octoflow run main.flow

print("Hello from OctoFlow!")
"#;
