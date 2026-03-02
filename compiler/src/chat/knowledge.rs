//! Knowledge tree — embedded L0/L1/L2/L3 content for context-aware prompting.
//!
//! 169 files from the Enhancement team's specs/context-engine/ directory,
//! compiled into the binary via include_str!(). The tree has four levels:
//!
//! - **L0**: Core syntax (~439 tokens, always in context)
//! - **L1**: Domain overviews (~50-75 tokens each, 18 domains)
//! - **L2**: Module signatures (~50-336 tokens each, 100 modules)
//! - **L3**: Function examples (~200-400 tokens each, 50 examples)

/// L0 — Core OctoFlow syntax reference. Always included in the system prompt.
pub const L0_CORE: &str = include_str!("knowledge/L0-core.md");

/// All 18 domain names.
pub const DOMAINS: &[&str] = &[
    "ai", "collections", "compiler", "crypto", "data", "db", "devops",
    "gpu", "gui", "loom", "media", "ml", "science", "stats", "string",
    "sys", "terminal", "web",
];

/// L1 domain overviews: (domain_name, content).
static L1_ENTRIES: &[(&str, &str)] = &[
    ("ai", include_str!("knowledge/L1/L1-ai.md")),
    ("collections", include_str!("knowledge/L1/L1-collections.md")),
    ("compiler", include_str!("knowledge/L1/L1-compiler.md")),
    ("crypto", include_str!("knowledge/L1/L1-crypto.md")),
    ("data", include_str!("knowledge/L1/L1-data.md")),
    ("db", include_str!("knowledge/L1/L1-db.md")),
    ("devops", include_str!("knowledge/L1/L1-devops.md")),
    ("gpu", include_str!("knowledge/L1/L1-gpu.md")),
    ("gui", include_str!("knowledge/L1/L1-gui.md")),
    ("loom", include_str!("knowledge/L1/L1-loom.md")),
    ("media", include_str!("knowledge/L1/L1-media.md")),
    ("ml", include_str!("knowledge/L1/L1-ml.md")),
    ("science", include_str!("knowledge/L1/L1-science.md")),
    ("stats", include_str!("knowledge/L1/L1-stats.md")),
    ("string", include_str!("knowledge/L1/L1-string.md")),
    ("sys", include_str!("knowledge/L1/L1-sys.md")),
    ("terminal", include_str!("knowledge/L1/L1-terminal.md")),
    ("web", include_str!("knowledge/L1/L1-web.md")),
];

/// L2 module signatures: (module_key, domain, content).
static L2_ENTRIES: &[(&str, &str, &str)] = &[
    ("ai-chat", "ai", include_str!("knowledge/L2/L2-ai-chat.md")),
    ("ai-gguf", "ai", include_str!("knowledge/L2/L2-ai-gguf.md")),
    ("ai-sampling", "ai", include_str!("knowledge/L2/L2-ai-sampling.md")),
    ("ai-tokenizer", "ai", include_str!("knowledge/L2/L2-ai-tokenizer.md")),
    ("collections-array_utils", "collections", include_str!("knowledge/L2/L2-collections-array_utils.md")),
    ("collections-collections", "collections", include_str!("knowledge/L2/L2-collections-collections.md")),
    ("collections-graph", "collections", include_str!("knowledge/L2/L2-collections-graph.md")),
    ("collections-heap", "collections", include_str!("knowledge/L2/L2-collections-heap.md")),
    ("collections-queue", "collections", include_str!("knowledge/L2/L2-collections-queue.md")),
    ("collections-sort", "collections", include_str!("knowledge/L2/L2-collections-sort.md")),
    ("collections-stack", "collections", include_str!("knowledge/L2/L2-collections-stack.md")),
    ("crypto-base64", "crypto", include_str!("knowledge/L2/L2-crypto-base64.md")),
    ("crypto-encoding", "crypto", include_str!("knowledge/L2/L2-crypto-encoding.md")),
    ("crypto-hash", "crypto", include_str!("knowledge/L2/L2-crypto-hash.md")),
    ("crypto-hex", "crypto", include_str!("knowledge/L2/L2-crypto-hex.md")),
    ("crypto-random", "crypto", include_str!("knowledge/L2/L2-crypto-random.md")),
    ("data-csv", "data", include_str!("knowledge/L2/L2-data-csv.md")),
    ("data-io", "data", include_str!("knowledge/L2/L2-data-io.md")),
    ("data-io_root", "data", include_str!("knowledge/L2/L2-data-io_root.md")),
    ("data-json", "data", include_str!("knowledge/L2/L2-data-json.md")),
    ("data-pipeline", "data", include_str!("knowledge/L2/L2-data-pipeline.md")),
    ("data-transform", "data", include_str!("knowledge/L2/L2-data-transform.md")),
    ("data-validate", "data", include_str!("knowledge/L2/L2-data-validate.md")),
    ("db-core", "db", include_str!("knowledge/L2/L2-db-core.md")),
    ("db-query", "db", include_str!("knowledge/L2/L2-db-query.md")),
    ("db-schema", "db", include_str!("knowledge/L2/L2-db-schema.md")),
    ("devops-config", "devops", include_str!("knowledge/L2/L2-devops-config.md")),
    ("devops-fs", "devops", include_str!("knowledge/L2/L2-devops-fs.md")),
    ("devops-log", "devops", include_str!("knowledge/L2/L2-devops-log.md")),
    ("devops-process", "devops", include_str!("knowledge/L2/L2-devops-process.md")),
    ("devops-template", "devops", include_str!("knowledge/L2/L2-devops-template.md")),
    ("gpu-builtins", "gpu", include_str!("knowledge/L2/L2-gpu-builtins.md")),
    ("gui-buffer_view", "gui", include_str!("knowledge/L2/L2-gui-buffer_view.md")),
    ("gui-canvas", "gui", include_str!("knowledge/L2/L2-gui-canvas.md")),
    ("gui-ecs", "gui", include_str!("knowledge/L2/L2-gui-ecs.md")),
    ("gui-gui_core", "gui", include_str!("knowledge/L2/L2-gui-gui_core.md")),
    ("gui-layout", "gui", include_str!("knowledge/L2/L2-gui-layout.md")),
    ("gui-physics2d", "gui", include_str!("knowledge/L2/L2-gui-physics2d.md")),
    ("gui-plot", "gui", include_str!("knowledge/L2/L2-gui-plot.md")),
    ("gui-render", "gui", include_str!("knowledge/L2/L2-gui-render.md")),
    ("gui-theme", "gui", include_str!("knowledge/L2/L2-gui-theme.md")),
    ("gui-widgets", "gui", include_str!("knowledge/L2/L2-gui-widgets.md")),
    ("gui-window", "gui", include_str!("knowledge/L2/L2-gui-window.md")),
    ("loom-data_composite", "loom", include_str!("knowledge/L2/L2-loom-data_composite.md")),
    ("loom-ops", "loom", include_str!("knowledge/L2/L2-loom-ops.md")),
    ("loom-patterns", "loom", include_str!("knowledge/L2/L2-loom-patterns.md")),
    ("loom-runtime", "loom", include_str!("knowledge/L2/L2-loom-runtime.md")),
    ("media-avi", "media", include_str!("knowledge/L2/L2-media-avi.md")),
    ("media-bmp", "media", include_str!("knowledge/L2/L2-media-bmp.md")),
    ("media-gif", "media", include_str!("knowledge/L2/L2-media-gif.md")),
    ("media-gif_encode", "media", include_str!("knowledge/L2/L2-media-gif_encode.md")),
    ("media-gif_encode_gpu", "media", include_str!("knowledge/L2/L2-media-gif_encode_gpu.md")),
    ("media-h264", "media", include_str!("knowledge/L2/L2-media-h264.md")),
    ("media-image_filter", "media", include_str!("knowledge/L2/L2-media-image_filter.md")),
    ("media-image_util", "media", include_str!("knowledge/L2/L2-media-image_util.md")),
    ("media-mp4", "media", include_str!("knowledge/L2/L2-media-mp4.md")),
    ("media-ttf", "media", include_str!("knowledge/L2/L2-media-ttf.md")),
    ("media-wav", "media", include_str!("knowledge/L2/L2-media-wav.md")),
    ("ml-classify", "ml", include_str!("knowledge/L2/L2-ml-classify.md")),
    ("ml-cluster", "ml", include_str!("knowledge/L2/L2-ml-cluster.md")),
    ("ml-ensemble", "ml", include_str!("knowledge/L2/L2-ml-ensemble.md")),
    ("ml-linalg", "ml", include_str!("knowledge/L2/L2-ml-linalg.md")),
    ("ml-metrics", "ml", include_str!("knowledge/L2/L2-ml-metrics.md")),
    ("ml-nn", "ml", include_str!("knowledge/L2/L2-ml-nn.md")),
    ("ml-preprocess", "ml", include_str!("knowledge/L2/L2-ml-preprocess.md")),
    ("ml-regression", "ml", include_str!("knowledge/L2/L2-ml-regression.md")),
    ("ml-tree", "ml", include_str!("knowledge/L2/L2-ml-tree.md")),
    ("science-calculus", "science", include_str!("knowledge/L2/L2-science-calculus.md")),
    ("science-constants", "science", include_str!("knowledge/L2/L2-science-constants.md")),
    ("science-interpolate", "science", include_str!("knowledge/L2/L2-science-interpolate.md")),
    ("science-math", "science", include_str!("knowledge/L2/L2-science-math.md")),
    ("science-matrix", "science", include_str!("knowledge/L2/L2-science-matrix.md")),
    ("science-optimize", "science", include_str!("knowledge/L2/L2-science-optimize.md")),
    ("science-physics", "science", include_str!("knowledge/L2/L2-science-physics.md")),
    ("science-signal", "science", include_str!("knowledge/L2/L2-science-signal.md")),
    ("stats-correlation", "stats", include_str!("knowledge/L2/L2-stats-correlation.md")),
    ("stats-descriptive", "stats", include_str!("knowledge/L2/L2-stats-descriptive.md")),
    ("stats-distribution", "stats", include_str!("knowledge/L2/L2-stats-distribution.md")),
    ("stats-hypothesis", "stats", include_str!("knowledge/L2/L2-stats-hypothesis.md")),
    ("stats-math_ext", "stats", include_str!("knowledge/L2/L2-stats-math_ext.md")),
    ("stats-risk", "stats", include_str!("knowledge/L2/L2-stats-risk.md")),
    ("stats-timeseries", "stats", include_str!("knowledge/L2/L2-stats-timeseries.md")),
    ("string-format", "string", include_str!("knowledge/L2/L2-string-format.md")),
    ("string-regex", "string", include_str!("knowledge/L2/L2-string-regex.md")),
    ("string-string", "string", include_str!("knowledge/L2/L2-string-string.md")),
    ("sys-args", "sys", include_str!("knowledge/L2/L2-sys-args.md")),
    ("sys-datetime", "sys", include_str!("knowledge/L2/L2-sys-datetime.md")),
    ("sys-env", "sys", include_str!("knowledge/L2/L2-sys-env.md")),
    ("sys-memory", "sys", include_str!("knowledge/L2/L2-sys-memory.md")),
    ("sys-platform", "sys", include_str!("knowledge/L2/L2-sys-platform.md")),
    ("sys-timer", "sys", include_str!("knowledge/L2/L2-sys-timer.md")),
    ("terminal-digits", "terminal", include_str!("knowledge/L2/L2-terminal-digits.md")),
    ("terminal-halfblock", "terminal", include_str!("knowledge/L2/L2-terminal-halfblock.md")),
    ("terminal-kitty", "terminal", include_str!("knowledge/L2/L2-terminal-kitty.md")),
    ("terminal-render", "terminal", include_str!("knowledge/L2/L2-terminal-render.md")),
    ("terminal-sixel", "terminal", include_str!("knowledge/L2/L2-terminal-sixel.md")),
    ("web-http", "web", include_str!("knowledge/L2/L2-web-http.md")),
    ("web-json_util", "web", include_str!("knowledge/L2/L2-web-json_util.md")),
    ("web-server", "web", include_str!("knowledge/L2/L2-web-server.md")),
    ("web-url", "web", include_str!("knowledge/L2/L2-web-url.md")),
];

/// L3 function examples: (function_name, content).
static L3_ENTRIES: &[(&str, &str)] = &[
    ("csv_filter", include_str!("knowledge/L3/L3-csv_filter.md")),
    ("filter", include_str!("knowledge/L3/L3-filter.md")),
    ("gpu_add", include_str!("knowledge/L3/L3-gpu_add.md")),
    ("gpu_fill", include_str!("knowledge/L3/L3-gpu_fill.md")),
    ("gpu_matmul", include_str!("knowledge/L3/L3-gpu_matmul.md")),
    ("gpu_mean", include_str!("knowledge/L3/L3-gpu_mean.md")),
    ("gpu_random", include_str!("knowledge/L3/L3-gpu_random.md")),
    ("gpu_scale", include_str!("knowledge/L3/L3-gpu_scale.md")),
    ("gpu_sum", include_str!("knowledge/L3/L3-gpu_sum.md")),
    ("graph_add_edge", include_str!("knowledge/L3/L3-graph_add_edge.md")),
    ("heap_push", include_str!("knowledge/L3/L3-heap_push.md")),
    ("http_get", include_str!("knowledge/L3/L3-http_get.md")),
    ("http_listen", include_str!("knowledge/L3/L3-http_listen.md")),
    ("http_post", include_str!("knowledge/L3/L3-http_post.md")),
    ("impute_missing", include_str!("knowledge/L3/L3-impute_missing.md")),
    ("json_parse", include_str!("knowledge/L3/L3-json_parse.md")),
    ("json_stringify", include_str!("knowledge/L3/L3-json_stringify.md")),
    ("kmeans", include_str!("knowledge/L3/L3-kmeans.md")),
    ("knn_predict", include_str!("knowledge/L3/L3-knn_predict.md")),
    ("linear_regression", include_str!("knowledge/L3/L3-linear_regression.md")),
    ("list_dir", include_str!("knowledge/L3/L3-list_dir.md")),
    ("loom_boot", include_str!("knowledge/L3/L3-loom_boot.md")),
    ("loom_dispatch_jit", include_str!("knowledge/L3/L3-loom_dispatch_jit.md")),
    ("map", include_str!("knowledge/L3/L3-map.md")),
    ("mean", include_str!("knowledge/L3/L3-mean.md")),
    ("normal_pdf", include_str!("knowledge/L3/L3-normal_pdf.md")),
    ("normalize", include_str!("knowledge/L3/L3-normalize.md")),
    ("now_ms", include_str!("knowledge/L3/L3-now_ms.md")),
    ("parse_args", include_str!("knowledge/L3/L3-parse_args.md")),
    ("pearson", include_str!("knowledge/L3/L3-pearson.md")),
    ("plot_create", include_str!("knowledge/L3/L3-plot_create.md")),
    ("print", include_str!("knowledge/L3/L3-print.md")),
    ("query_string", include_str!("knowledge/L3/L3-query_string.md")),
    ("re_match", include_str!("knowledge/L3/L3-re_match.md")),
    ("read_csv", include_str!("knowledge/L3/L3-read_csv.md")),
    ("read_file", include_str!("knowledge/L3/L3-read_file.md")),
    ("send_response_cors", include_str!("knowledge/L3/L3-send_response_cors.md")),
    ("sharpe_ratio", include_str!("knowledge/L3/L3-sharpe_ratio.md")),
    ("sma", include_str!("knowledge/L3/L3-sma.md")),
    ("sort_array", include_str!("knowledge/L3/L3-sort_array.md")),
    ("stack_push", include_str!("knowledge/L3/L3-stack_push.md")),
    ("struct", include_str!("knowledge/L3/L3-struct.md")),
    ("t_test_one_sample", include_str!("knowledge/L3/L3-t_test_one_sample.md")),
    ("train_test_split", include_str!("knowledge/L3/L3-train_test_split.md")),
    ("try", include_str!("knowledge/L3/L3-try.md")),
    ("url_encode", include_str!("knowledge/L3/L3-url_encode.md")),
    ("walk_dir", include_str!("knowledge/L3/L3-walk_dir.md")),
    ("window_open", include_str!("knowledge/L3/L3-window_open.md")),
    ("write_csv", include_str!("knowledge/L3/L3-write_csv.md")),
    ("write_file", include_str!("knowledge/L3/L3-write_file.md")),
];

/// Get the L1 overview for a domain.
pub fn get_l1(domain: &str) -> Option<&'static str> {
    L1_ENTRIES.iter()
        .find(|(d, _)| *d == domain)
        .map(|(_, content)| *content)
}

/// Get all L2 modules for a domain. Returns vec of (module_key, content).
pub fn get_l2_for_domain(domain: &str) -> Vec<(&'static str, &'static str)> {
    L2_ENTRIES.iter()
        .filter(|(_, d, _)| *d == domain)
        .map(|(key, _, content)| (*key, *content))
        .collect()
}

/// Get a specific L2 module by key (e.g. "web-http").
pub fn get_l2(module_key: &str) -> Option<&'static str> {
    L2_ENTRIES.iter()
        .find(|(key, _, _)| *key == module_key)
        .map(|(_, _, content)| *content)
}

/// Get a specific L3 example by function name.
pub fn get_l3(function: &str) -> Option<&'static str> {
    L3_ENTRIES.iter()
        .find(|(f, _)| *f == function)
        .map(|(_, content)| *content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l0_core_not_empty() {
        assert!(!L0_CORE.is_empty());
        assert!(L0_CORE.contains("OctoFlow"));
    }

    #[test]
    fn test_domains_count() {
        assert_eq!(DOMAINS.len(), 18);
        assert!(DOMAINS.contains(&"web"));
        assert!(DOMAINS.contains(&"gpu"));
        assert!(DOMAINS.contains(&"ml"));
    }

    #[test]
    fn test_l1_lookup() {
        assert!(get_l1("web").is_some());
        assert!(get_l1("gpu").is_some());
        assert!(get_l1("nonexistent").is_none());
        let web = get_l1("web").unwrap();
        assert!(web.contains("http") || web.contains("HTTP"));
    }

    #[test]
    fn test_l2_for_domain() {
        let web_modules = get_l2_for_domain("web");
        assert!(web_modules.len() >= 3); // http, server, url, json_util
        let keys: Vec<&str> = web_modules.iter().map(|(k, _)| *k).collect();
        assert!(keys.contains(&"web-http"));
    }

    #[test]
    fn test_l2_lookup() {
        assert!(get_l2("web-http").is_some());
        assert!(get_l2("gpu-builtins").is_some());
        assert!(get_l2("nonexistent").is_none());
    }

    #[test]
    fn test_l3_lookup() {
        assert!(get_l3("http_get").is_some());
        assert!(get_l3("print").is_some());
        assert!(get_l3("nonexistent").is_none());
    }

    #[test]
    fn test_l1_entry_count() {
        assert_eq!(L1_ENTRIES.len(), 18);
    }

    #[test]
    fn test_l2_entry_count() {
        assert_eq!(L2_ENTRIES.len(), 100);
    }

    #[test]
    fn test_l3_entry_count() {
        assert_eq!(L3_ENTRIES.len(), 50);
    }
}
