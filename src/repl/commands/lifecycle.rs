//! Server lifecycle REPL commands: /provenance, /restart, /ctx, /model, /trio, /local.

use std::io::{self, Write as _};
use std::path::PathBuf;

use super::*;

#[derive(Debug, Clone)]
pub(crate) struct ModelSwitchReport {
    pub(crate) model_id: String,
    pub(crate) notes: Vec<String>,
}

impl ModelSwitchReport {
    fn new(model_id: String) -> Self {
        Self {
            model_id,
            notes: Vec::new(),
        }
    }

    fn note(&mut self, text: impl Into<String>) {
        self.notes.push(text.into());
    }
}

impl ReplContext {
    /// /provenance — toggle provenance display on/off.
    pub(super) fn cmd_provenance(&mut self) {
        let was_enabled = {
            let core = self.core_handle.swappable();
            core.provenance_config.enabled
        };
        // Toggle by rebuilding core with modified config.
        let mut toggled_config = self.config.clone();
        toggled_config.provenance.enabled = !was_enabled;
        let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
            Some(self.config.agents.defaults.lms_main_model.as_str())
        } else {
            self.current_model_path.file_name().and_then(|n| n.to_str())
        };
        cli::rebuild_core(
            &self.core_handle,
            &toggled_config,
            &self.srv.local_port,
            model_name,
            None,
            None,
            None,
            // migrated from swappable().is_local — phase 09-03
            self.core_handle.swappable().mode().is_local(),
        );
        self.agent_loop = cli::create_agent_loop(
            self.core_handle.clone(),
            &toggled_config,
            Some(self.cron_service.clone()),
            self.email_config.clone(),
            Some(self.display_tx.clone()),
            self.health_registry.clone(),
        );
        if !was_enabled {
            println!(
                "\n  Provenance \x1b[32menabled\x1b[0m (tool calls visible, audit logging on)\n"
            );
        } else {
            println!("\n  Provenance \x1b[33mdisabled\x1b[0m\n");
        }
    }

    /// Point config at a running Higgs instance and refresh the model name.
    ///
    /// Shared by `/restart`, `/m` model switch, and auto-start so the
    /// "update local_api_base + refresh model name" sequence lives in one place.
    /// Returns the model name if the server responded, or None.
    async fn apply_higgs_endpoint(&mut self, port: u16) -> Option<String> {
        self.config.agents.defaults.local_api_base = format!("http://127.0.0.1:{port}/v1");
        // Prefer the model the user configured; only fall back to whatever Higgs
        // serves first when that model isn't among the served set. Avoids adopting
        // a sibling model (e.g. a multi-model Higgs) that triggers a 404 later.
        let preferred = if !self.config.agents.defaults.lms_main_model.is_empty() {
            self.config.agents.defaults.lms_main_model.clone()
        } else {
            self.config.agents.defaults.local_model.clone()
        };
        if let Some(name) = crate::higgs::resolve_served_model(port, &preferred).await {
            self.config.agents.defaults.lms_main_model = name.clone();
            self.config.agents.defaults.local_model = name.clone();
            Some(name)
        } else {
            None
        }
    }

    /// /restart — restart local servers and reload models.
    pub async fn cmd_restart(&mut self) {
        // Higgs managed sidecar: stop → restart with current model dir.
        if self.srv.engine == super::super::InferenceEngine::Higgs {
            let port = self.config.agents.defaults.higgs_port;
            match crate::higgs::resolve_model_dir(&self.config) {
                Ok(model_dir) => {
                    if let Some(bin) = crate::higgs::find_binary() {
                        print!("  Restarting Higgs server... ");
                        io::stdout().flush().ok();
                        match crate::higgs::server_restart(
                            &bin,
                            port,
                            &model_dir,
                            &self.config.agents.defaults.local_model,
                        )
                        .await
                        {
                            Ok(crate::higgs::StartResult::Ready) => {
                                println!("{}OK{}", tui::GREEN, tui::RESET);
                                if let Some(name) = self.apply_higgs_endpoint(port).await {
                                    println!("  Model: {name}");
                                }
                            }
                            Ok(crate::higgs::StartResult::Loading { pid, port: _ }) => {
                                println!(
                                    "{}LOADING{} (pid {} still warming up)",
                                    tui::YELLOW,
                                    tui::RESET,
                                    pid
                                );
                                self.config.agents.defaults.local_api_base =
                                    format!("http://127.0.0.1:{port}/v1");
                                self.config.agents.defaults.skip_jit_gate = true;
                            }
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    } else {
                        println!(
                            "  {}Higgs binary not found.{} Install with: cargo install higgs",
                            tui::YELLOW,
                            tui::RESET,
                        );
                    }
                }
                Err(e) => println!("  {}Config error:{} {e}", tui::YELLOW, tui::RESET),
            }
            self.apply_and_rebuild();
            println!(
                "  {}{}Rebuilt{} agent core.",
                tui::BOLD,
                tui::GREEN,
                tui::RESET
            );
            return;
        }

        if self.srv.lms_managed {
            if let Some(ref bin) = self.srv.lms_binary.clone() {
                let lms_port = self.config.agents.defaults.lms_port;

                // Restart LMS server
                print!("  Restarting LM Studio server... ");
                io::stdout().flush().ok();
                crate::lms::server_stop(bin).ok();
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                match crate::lms::server_start(bin, lms_port).await {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => {
                        println!("{}FAILED: {}{}", tui::RED, e, tui::RESET);
                        return;
                    }
                }

                // Reload main model
                let main_model = if !self.config.agents.defaults.lms_main_model.is_empty() {
                    self.config.agents.defaults.lms_main_model.clone()
                } else {
                    self.config.agents.defaults.local_model.clone()
                };
                let main_ctx = Some(self.config.agents.defaults.local_max_context_tokens);
                print!("  Loading {}... ", main_model);
                io::stdout().flush().ok();
                match crate::lms::load_model(
                    "",
                    lms_port,
                    &main_model,
                    main_ctx,
                    self.config.timeouts.lms_load_secs,
                )
                .await
                {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                }

                // Reload trio models if enabled
                if self.config.trio.enabled {
                    if !self.config.trio.router_model.is_empty() {
                        print!("  Loading {}... ", self.config.trio.router_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model(
                            "",
                            lms_port,
                            &self.config.trio.router_model,
                            Some(self.config.trio.router_ctx_tokens),
                            self.config.timeouts.lms_load_secs,
                        )
                        .await
                        {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    }
                    if !self.config.trio.specialist_model.is_empty() {
                        print!("  Loading {}... ", self.config.trio.specialist_model);
                        io::stdout().flush().ok();
                        match crate::lms::load_model(
                            "",
                            lms_port,
                            &self.config.trio.specialist_model,
                            Some(self.config.trio.specialist_ctx_tokens),
                            self.config.timeouts.lms_load_secs,
                        )
                        .await
                        {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    }
                }

                // Update URL in case host/port changed (only if user has not set an explicit base)
                if self.config.agents.defaults.local_api_base.is_empty() {
                    let lms_host = crate::lms::api_host();
                    self.config.agents.defaults.local_api_base =
                        format!("http://{}:{}/v1", lms_host, lms_port);
                }
            }
        }

        self.apply_and_rebuild();
        println!(
            "  {}{}Rebuilt{} agent core.",
            tui::BOLD,
            tui::GREEN,
            tui::RESET
        );
    }

    /// /ctx [size] — show or set context size for the main model.
    pub(super) async fn cmd_ctx(&mut self, arg: &str) {
        // migrated from swappable().is_local — phase 09-03
        if !self.core_handle.swappable().mode().is_local() {
            println!(
                "\n  {}Not in local mode — use /local first{}\n",
                tui::DIM,
                tui::RESET
            );
            return;
        }

        let new_ctx: usize = match super::super::parse_ctx_arg(arg) {
            Ok(Some(n)) => n,
            Ok(None) => {
                let auto = server::compute_optimal_context_size(&self.current_model_path);
                let current = self.config.agents.defaults.local_max_context_tokens;
                println!("\n  Current: {}K", current / 1024);
                println!("  Auto-detected optimal: {}K", auto / 1024);
                if self.config.trio.enabled {
                    let budget = self.compute_current_vram_budget();
                    let total_gb = budget.total_vram_bytes as f64 / 1e9;
                    let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
                    let status = if budget.fits { "OK" } else { "OVER" };
                    println!("  VRAM: {:.1} / {:.1} GB [{}]", total_gb, limit_gb, status);
                }
                println!("\n  Usage: /ctx <size>  e.g. /ctx 32K or /ctx 32768\n");
                return;
            }
            Err(msg) => {
                println!("\n  {}\n", msg);
                println!("  Usage: /ctx <size>  e.g. /ctx 32K or /ctx 32768\n");
                return;
            }
        };

        // Apply context change
        self.config.agents.defaults.local_max_context_tokens = new_ctx;

        // Persist to disk
        let mut disk_cfg = load_config(None);
        disk_cfg.agents.defaults.local_max_context_tokens = new_ctx;
        save_config(&disk_cfg, None);

        // MLX mode: no LMS server to reload — just rebuild with new context cap.

        // Reload model in LMS with new context
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
                self.config.agents.defaults.lms_main_model.clone()
            } else {
                self.config.agents.defaults.local_model.clone()
            };
            if !model_name.is_empty() {
                print!(
                    "  Reloading {} with {}K context... ",
                    model_name,
                    new_ctx / 1024
                );
                io::stdout().flush().ok();
                match crate::lms::reload_model_with_context(
                    "",
                    lms_port,
                    &model_name,
                    new_ctx,
                    self.config.timeouts.lms_load_secs,
                    self.config.timeouts.lms_unload_secs,
                )
                .await
                {
                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                    Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                }
            }
        } else if !self.config.agents.defaults.local_api_base.is_empty() {
            // Remote LMS peer: reload model with new context on that server
            #[cfg(feature = "cluster")]
            let is_lms = self.is_remote_lms_peer().await;
            #[cfg(not(feature = "cluster"))]
            let is_lms = Self::extract_endpoint_port(&self.config.agents.defaults.local_api_base)
                == Some(1234);

            if is_lms {
                if let Some(port) =
                    Self::extract_endpoint_port(&self.config.agents.defaults.local_api_base)
                {
                    let remote_host =
                        super::super::extract_url_host(&self.config.agents.defaults.local_api_base);
                    let model_name = if !self.config.agents.defaults.lms_main_model.is_empty() {
                        self.config.agents.defaults.lms_main_model.clone()
                    } else {
                        self.config.agents.defaults.local_model.clone()
                    };
                    if !model_name.is_empty() {
                        print!(
                            "  Reloading {} with {}K context on remote LMS... ",
                            model_name,
                            new_ctx / 1024
                        );
                        io::stdout().flush().ok();
                        match crate::lms::reload_model_with_context(
                            &remote_host,
                            port,
                            &model_name,
                            new_ctx,
                            self.config.timeouts.lms_load_secs,
                            self.config.timeouts.lms_unload_secs,
                        )
                        .await
                        {
                            Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                            Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                        }
                    }
                }
            }
        }

        // Warn if VRAM budget exceeded
        if self.config.trio.enabled {
            let budget = self.compute_current_vram_budget();
            if !budget.fits {
                println!(
                    "  \x1b[33mWarning:\x1b[0m Total VRAM ({:.1} GB) exceeds limit ({:.1} GB).",
                    budget.total_vram_bytes as f64 / 1e9,
                    budget.effective_limit_bytes as f64 / 1e9,
                );
                println!("  Reduce context sizes or switch to smaller models.");
            }
        }

        self.apply_and_rebuild();
        println!(
            "\n  Context size set to {}{}K{}.\n",
            tui::BOLD,
            new_ctx / 1024,
            tui::RESET,
        );
    }

    /// Apply one model picker entry without doing any terminal I/O.
    ///
    /// The classic REPL and ratatui picker share this path so model switching
    /// has one set of load/unload, persistence, and rebuild rules.
    pub(crate) async fn apply_model_entry(
        &mut self,
        selected: ModelEntry,
    ) -> Result<ModelSwitchReport, String> {
        let selected_id = selected.id.clone();
        let mut report = ModelSwitchReport::new(selected_id.clone());

        match selected.source {
            ModelSource::LocalLms { port } => {
                let prev_model = self.config.agents.defaults.lms_main_model.clone();
                if !prev_model.is_empty() && prev_model != selected_id {
                    match crate::lms::unload_model(
                        "",
                        port,
                        &prev_model,
                        self.config.timeouts.lms_unload_secs,
                    )
                    .await
                    {
                        Ok(()) => report.note(format!("unloaded {prev_model}")),
                        Err(e) => report.note(format!("warning: unload {prev_model}: {e}")),
                    }
                }

                let ctx = Some(self.config.agents.defaults.local_max_context_tokens);
                crate::lms::load_model(
                    "",
                    port,
                    &selected_id,
                    ctx,
                    self.config.timeouts.lms_load_secs,
                )
                .await
                .map_err(|e| format!("failed to load {selected_id}: {e}"))?;
                report.note(format!("loaded {selected_id}"));

                self.config.agents.defaults.local_model = selected_id.clone();
                self.config.agents.defaults.lms_main_model = selected_id.clone();
                if !crate::config::schema::is_higgs_backend(
                    &self.config.agents.defaults.local_backend,
                ) {
                    self.config.agents.defaults.local_backend = "lmstudio".to_string();
                }
                self.current_model_path = PathBuf::from(&selected_id);

                let mut disk_cfg = load_config(None);
                disk_cfg.agents.defaults.local_model = selected_id.clone();
                disk_cfg.agents.defaults.lms_main_model = selected_id.clone();
                if !crate::config::schema::is_higgs_backend(&disk_cfg.agents.defaults.local_backend)
                {
                    disk_cfg.agents.defaults.local_backend = "lmstudio".to_string();
                }
                save_config(&disk_cfg, None);
                self.apply_and_rebuild();
            }
            ModelSource::Remote {
                endpoint,
                #[cfg(feature = "cluster")]
                peer_type,
                #[cfg(not(feature = "cluster"))]
                    peer_type: _,
            } => {
                #[cfg(feature = "cluster")]
                let is_lms = peer_type == crate::cluster::state::PeerType::LMStudio;
                #[cfg(not(feature = "cluster"))]
                let is_lms = false;

                if is_lms {
                    let Some(port) = Self::extract_endpoint_port(&endpoint) else {
                        return Err(format!(
                            "could not parse LM Studio endpoint port from {endpoint}"
                        ));
                    };

                    let remote_host = super::super::extract_url_host(&endpoint);
                    let prev_model = self.config.agents.defaults.lms_main_model.clone();
                    if !prev_model.is_empty() && prev_model != selected_id {
                        match crate::lms::unload_model(
                            &remote_host,
                            port,
                            &prev_model,
                            self.config.timeouts.lms_unload_secs,
                        )
                        .await
                        {
                            Ok(()) => report.note(format!("unloaded {prev_model}")),
                            Err(e) => report.note(format!("warning: unload {prev_model}: {e}")),
                        }
                    }

                    let ctx = Some(self.config.agents.defaults.local_max_context_tokens);
                    crate::lms::load_model(
                        &remote_host,
                        port,
                        &selected_id,
                        ctx,
                        self.config.timeouts.lms_load_secs,
                    )
                    .await
                    .map_err(|e| format!("failed to load {selected_id}: {e}"))?;
                    report.note(format!("loaded {selected_id}"));
                }

                self.config.agents.defaults.local_api_base = endpoint;
                self.config.agents.defaults.lms_main_model = selected_id.clone();
                self.config.agents.defaults.local_model = selected_id.clone();
                if !crate::config::schema::is_higgs_backend(
                    &self.config.agents.defaults.local_backend,
                ) && self.config.agents.defaults.local_backend == "mlx"
                {
                    self.config.agents.defaults.local_backend = if is_lms {
                        "lmstudio".to_string()
                    } else {
                        "omlx".to_string()
                    };
                }
                if !is_lms {
                    self.config.agents.defaults.skip_jit_gate = true;
                }
                self.srv.lms_managed = false;
                self.current_model_path = PathBuf::from(&selected_id);
                self.persist_local_config();
                self.apply_and_rebuild_with(true);
            }
            ModelSource::File { path } => {
                self.current_model_path = path.clone();
                let name = path.file_name().unwrap().to_string_lossy().to_string();
                self.config.agents.defaults.local_model = name.clone();
                let mut disk_cfg = load_config(None);
                disk_cfg.agents.defaults.local_model = name;
                save_config(&disk_cfg, None);
                self.apply_and_rebuild();
            }
            ModelSource::Omlx { endpoint } => {
                self.config.agents.defaults.local_api_base = endpoint;
                self.config.agents.defaults.local_model = selected_id.clone();
                self.config.agents.defaults.lms_main_model = selected_id.clone();
                self.config.agents.defaults.local_backend = "omlx".to_string();
                self.srv.lms_managed = false;
                self.current_model_path = PathBuf::from(&selected_id);
                self.persist_local_config();
                self.apply_and_rebuild_with(true);
                report.note(format!(
                    "switched to {selected_id} (oMLX will load on first request)"
                ));
            }
            ModelSource::Higgs {
                endpoint,
                path,
                name,
            } => {
                self.config.agents.defaults.local_api_base = endpoint.clone();
                self.config.agents.defaults.local_backend = "higgs".to_string();
                self.config.agents.defaults.skip_jit_gate = true;
                self.srv.lms_managed = false;
                self.srv.engine = super::super::InferenceEngine::Higgs;

                if let Some(model_path) = path {
                    let loaded_name = crate::higgs::switch_runtime_model(
                        &endpoint,
                        &self.config.agents.defaults.local_api_key,
                        &model_path,
                        &name,
                        self.config.timeouts.lms_load_secs,
                    )
                    .await
                    .map_err(|e| format!("failed to switch Higgs to {selected_id}: {e}"))?;
                    report.note(format!("Higgs loaded {loaded_name}"));

                    self.config.agents.defaults.local_model = selected_id.clone();
                    self.config.agents.defaults.lms_main_model = "active".to_string();
                    self.config.agents.defaults.mlx_model_dir = Some(model_path.clone());
                    self.current_model_path = PathBuf::from(model_path.as_str());
                } else {
                    self.config.agents.defaults.local_model = selected_id.clone();
                    self.config.agents.defaults.lms_main_model = selected_id.clone();
                    self.current_model_path = PathBuf::from(&selected_id);
                    report.note(format!("routing to resident Higgs model {selected_id}"));
                }

                self.persist_local_config();
                self.apply_and_rebuild_with(true);
            }
        }

        self.core_handle
            .counters
            .reset_session_prompt_state(&self.session_id);

        if self.config.trio.enabled {
            let budget = self.compute_current_vram_budget();
            if !budget.fits {
                report.note(format!(
                    "warning: VRAM usage ({:.1} GB) exceeds limit ({:.1} GB); use /trio budget for details",
                    budget.total_vram_bytes as f64 / 1e9,
                    budget.effective_limit_bytes as f64 / 1e9,
                ));
            }
        }

        Ok(report)
    }

    /// /model [filter] — unified model picker across all sources.
    pub(super) async fn cmd_model(&mut self, filter: &str) {
        let has_cluster = {
            #[cfg(feature = "cluster")]
            {
                self.cluster_state.is_some()
            }
            #[cfg(not(feature = "cluster"))]
            {
                false
            }
        };

        // migrated from swappable().is_local — phase 09-03
        if !self.core_handle.swappable().mode().is_local() && !has_cluster {
            println!("\n  /model is only available in local mode. Use /local to switch.\n");
            return;
        }

        let all = self.collect_all_models().await;
        let filter_lower = filter.trim().to_lowercase();
        let entries: Vec<&ModelEntry> = if filter_lower.is_empty() {
            all.iter().collect()
        } else {
            all.iter()
                .filter(|e| e.id.to_lowercase().contains(&filter_lower))
                .collect()
        };

        if entries.is_empty() {
            if filter_lower.is_empty() {
                println!("\n  No models found.\n");
            } else {
                println!("\n  No models matching \"{}\".\n", filter);
            }
            return;
        }

        let selected = if !filter_lower.is_empty() {
            unique_direct_model_match(&entries, &filter_lower).cloned()
        } else {
            None
        };

        let selected = if let Some(selected) = selected {
            selected
        } else {
            // Group entries by source label for display
            println!();
            let mut current_group = String::new();
            for (i, entry) in entries.iter().enumerate() {
                let group = match &entry.source {
                    ModelSource::LocalLms { port } => {
                        format!("Local (LM Studio :{})", port)
                    }
                    ModelSource::Remote {
                        endpoint,
                        peer_type,
                    } => {
                        let short = crate::tui::shorten_url(endpoint)
                            .split('/')
                            .next()
                            .unwrap_or(endpoint);
                        #[cfg(feature = "cluster")]
                        {
                            format!("{} ({})", short, peer_type)
                        }
                        #[cfg(not(feature = "cluster"))]
                        {
                            let _ = peer_type;
                            format!("{}", short)
                        }
                    }
                    ModelSource::File { .. } => "~/models/".to_string(),
                    ModelSource::Omlx { ref endpoint } => {
                        let short = crate::tui::shorten_url(endpoint)
                            .split('/')
                            .next()
                            .unwrap_or(endpoint);
                        format!("oMLX ({})", short)
                    }
                    ModelSource::Higgs { ref endpoint, .. } => {
                        let short = crate::tui::shorten_url(endpoint)
                            .split('/')
                            .next()
                            .unwrap_or(endpoint);
                        format!("Higgs ({})", short)
                    }
                };
                if group != current_group {
                    if !current_group.is_empty() {
                        println!();
                    }
                    println!("  {}{}{}:", tui::DIM, group, tui::RESET);
                    current_group = group;
                }
                let marker = if entry.is_active {
                    format!(" {}(active){}", tui::GREEN, tui::RESET)
                } else if entry.is_loaded {
                    format!(" {}(loaded){}", tui::DIM, tui::RESET)
                } else {
                    String::new()
                };
                // For file entries, show size
                if let ModelSource::File { ref path } = entry.source {
                    let size_mb = std::fs::metadata(path)
                        .map(|m| m.len() / 1_048_576)
                        .unwrap_or(0);
                    println!("    [{}] {} ({} MB){}", i + 1, entry.id, size_mb, marker);
                } else {
                    println!("    [{}] {}{}", i + 1, entry.id, marker);
                }
            }

            // Prompt for selection
            let prompt = format!("\nSelect model [1-{}] or Enter to cancel: ", entries.len());
            let choice = match self.rl.as_mut().unwrap().readline(&prompt) {
                Ok(line) => line,
                Err(_) => return,
            };
            let choice = choice.trim();
            if choice.is_empty() {
                return;
            };
            let idx: usize = match choice.parse::<usize>() {
                Ok(n) if n >= 1 && n <= entries.len() => n - 1,
                _ => {
                    println!("  Invalid selection.\n");
                    return;
                }
            };
            entries[idx].clone()
        };

        println!("\n  Selected: {}", selected.id);
        match self.apply_model_entry(selected).await {
            Ok(report) => {
                for note in report.notes {
                    println!("  {note}");
                }
                println!(
                    "  {}Model switched to {}.{}\n",
                    tui::DIM,
                    report.model_id,
                    tui::RESET
                );
            }
            Err(e) => {
                println!("  {}FAILED:{} {}", tui::RED, tui::RESET, e);
                println!(
                    "  {}Model switch aborted; config was not changed.{}\n",
                    tui::DIM,
                    tui::RESET
                );
            }
        }
    }

    /// /trio — manage trio mode (router + specialist helpers).
    ///
    /// Subcommands:
    ///   /trio                      — toggle trio on/off
    ///   /trio status               — show current trio config
    ///   /trio budget               — show VRAM budget breakdown
    ///   /trio router               — pick router model from LM Studio
    ///   /trio specialist           — pick specialist model from LM Studio
    ///   /trio router temp 0.3      — set router temperature
    ///   /trio specialist ctx 8K    — set specialist context size
    ///   /trio router nothink       — toggle router no_think
    ///   /trio main nothink         — toggle main no_think
    ///   /trio cap 12               — set VRAM cap (GB)
    pub(super) async fn cmd_trio(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        match parts.as_slice() {
            ["status" | "s"] => self.cmd_trio_status().await,
            ["router" | "r"] => self.cmd_trio_pick_model("router").await,
            ["specialist" | "spec"] => self.cmd_trio_pick_model("specialist").await,
            ["budget" | "b"] => self.cmd_trio_budget().await,

            // Parameter subcommands
            ["router" | "r", "temp" | "temperature", val] => {
                self.cmd_trio_set_param("router", "temperature", val)
            }
            ["specialist" | "spec", "temp" | "temperature", val] => {
                self.cmd_trio_set_param("specialist", "temperature", val)
            }
            ["router" | "r", "ctx" | "context", val] => {
                self.cmd_trio_set_param("router", "ctx", val)
            }
            ["specialist" | "spec", "ctx" | "context", val] => {
                self.cmd_trio_set_param("specialist", "ctx", val)
            }
            ["router" | "r", "nothink" | "no_think"] => {
                self.cmd_trio_set_param("router", "no_think", "toggle")
            }
            ["main", "nothink" | "no_think"] => {
                self.cmd_trio_set_param("main", "no_think", "toggle")
            }
            ["cap" | "vram", val] => self.cmd_trio_set_param("trio", "vram_cap", val),

            [] => self.cmd_trio_toggle().await,
            _ => {
                println!("\n  Usage: /trio [subcommand]");
                println!("    /trio                      Toggle trio on/off");
                println!("    /trio status               Show current trio config");
                println!("    /trio budget               Show VRAM budget breakdown");
                println!("    /trio router               Pick router model");
                println!("    /trio specialist            Pick specialist model");
                println!("    /trio router temp 0.3       Set router temperature");
                println!("    /trio specialist temp 0.7   Set specialist temperature");
                println!("    /trio router ctx 4K         Set router context");
                println!("    /trio specialist ctx 8K     Set specialist context");
                println!("    /trio router nothink        Toggle router no_think");
                println!("    /trio main nothink          Toggle main no_think");
                println!("    /trio cap 12                Set VRAM cap (GB)\n");
            }
        }
    }

    /// Set a trio parameter (temperature, context, no_think, vram_cap).
    fn cmd_trio_set_param(&mut self, role: &str, param: &str, value: &str) {
        match apply_trio_param(&mut self.config, role, param, value) {
            Ok(desc) => {
                self.persist_trio_config();
                self.apply_and_rebuild();
                println!("\n  Set {}.\n", desc);
            }
            Err(msg) => {
                println!("\n  Error: {}\n", msg);
            }
        }
    }

    /// Show VRAM budget breakdown.
    async fn cmd_trio_budget(&self) {
        // migrated from swappable().is_local — phase 09-03
        if !self.core_handle.swappable().mode().is_local() {
            println!(
                "\n  {}Not in local mode — use /local first{}\n",
                tui::DIM,
                tui::RESET
            );
            return;
        }
        let budget = self.compute_current_vram_budget();
        println!("{}", format_vram_budget(&budget));
    }

    /// Toggle trio mode on/off.
    async fn cmd_trio_toggle(&mut self) {
        let was_enabled = self.config.trio.enabled;

        if was_enabled {
            trio_disable(&mut self.config);

            // Unload router + specialist from LMS (keep main loaded)
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                if !self.config.trio.router_model.is_empty() {
                    let _ = crate::lms::unload_model(
                        "",
                        lms_port,
                        &self.config.trio.router_model,
                        self.config.timeouts.lms_unload_secs,
                    )
                    .await;
                }
                if !self.config.trio.specialist_model.is_empty() {
                    let _ = crate::lms::unload_model(
                        "",
                        lms_port,
                        &self.config.trio.specialist_model,
                        self.config.timeouts.lms_unload_secs,
                    )
                    .await;
                }
            }

            self.persist_trio_config();
            self.apply_and_rebuild();
            println!("\n  Trio \x1b[33mdisabled\x1b[0m — single model (inline) mode.\n");
        } else {
            let needs_warning = trio_enable(&mut self.config);
            if needs_warning {
                println!("\n  \x1b[33mWarning:\x1b[0m Router or specialist model not configured.");
                println!("  Use /trio router and /trio specialist to pick models first.");
                println!("  Or set them in config.json under \"trio\".\n");
            }

            // Auto-compute optimal context sizes to fit VRAM budget
            // migrated from swappable().is_local — phase 09-03
            if self.core_handle.swappable().mode().is_local() {
                let budget = self.compute_current_vram_budget();
                if budget.fits {
                    // Apply computed context sizes
                    self.config.agents.defaults.local_max_context_tokens = budget.main_ctx;
                    if budget.router_ctx > 0 {
                        self.config.trio.router_ctx_tokens = budget.router_ctx;
                    }
                    if budget.specialist_ctx > 0 {
                        self.config.trio.specialist_ctx_tokens = budget.specialist_ctx;
                    }
                    let total_gb = budget.total_vram_bytes as f64 / 1e9;
                    let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
                    println!(
                        "  Auto-computed contexts: main={}K router={}K specialist={}K ({:.1}/{:.1} GB)",
                        budget.main_ctx / 1024,
                        budget.router_ctx / 1024,
                        budget.specialist_ctx / 1024,
                        total_gb,
                        limit_gb,
                    );
                } else {
                    println!(
                        "  \x1b[33mWarning:\x1b[0m Models may exceed VRAM ({:.1}/{:.1} GB).",
                        budget.total_vram_bytes as f64 / 1e9,
                        budget.effective_limit_bytes as f64 / 1e9,
                    );
                    println!("  Use /trio budget for details, /trio cap to adjust.");
                }
            }

            // Load router + specialist on LMS if available
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                if !self.config.trio.router_model.is_empty() {
                    print!("  Loading {}... ", self.config.trio.router_model);
                    io::stdout().flush().ok();
                    match crate::lms::load_model(
                        "",
                        lms_port,
                        &self.config.trio.router_model,
                        Some(self.config.trio.router_ctx_tokens),
                        self.config.timeouts.lms_load_secs,
                    )
                    .await
                    {
                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                    }
                }
                if !self.config.trio.specialist_model.is_empty() {
                    print!("  Loading {}... ", self.config.trio.specialist_model);
                    io::stdout().flush().ok();
                    match crate::lms::load_model(
                        "",
                        lms_port,
                        &self.config.trio.specialist_model,
                        Some(self.config.trio.specialist_ctx_tokens),
                        self.config.timeouts.lms_load_secs,
                    )
                    .await
                    {
                        Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                        Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
                    }
                }
            }

            self.persist_trio_config();
            self.apply_and_rebuild();
            println!("\n  Trio \x1b[32menabled\x1b[0m — main + router + specialist.\n");
        }
    }

    /// Show current trio configuration.
    async fn cmd_trio_status(&self) {
        let trio = &self.config.trio;
        let td = &self.config.tool_delegation;
        let enabled_label = if trio.enabled {
            format!("{}enabled{}", tui::GREEN, tui::RESET)
        } else {
            format!("{}disabled{}", tui::YELLOW, tui::RESET)
        };

        println!();
        println!("  {}TRIO{}       {}", tui::BOLD, tui::RESET, enabled_label);
        println!("  {}MODE{}       {:?}", tui::BOLD, tui::RESET, td.mode);

        // Main model
        let main = if !self.config.agents.defaults.lms_main_model.is_empty() {
            self.config.agents.defaults.lms_main_model.clone()
        } else if !self.config.agents.defaults.local_model.is_empty() {
            self.config.agents.defaults.local_model.clone()
        } else {
            "(default)".to_string()
        };
        println!(
            "  {}MAIN{}       {}{}{}",
            tui::BOLD,
            tui::RESET,
            tui::DIM,
            main,
            tui::RESET
        );

        // Router
        let router = if trio.router_model.is_empty() {
            "\x1b[33m(not set)\x1b[0m".to_string()
        } else {
            format!("{}{}{}", tui::DIM, trio.router_model, tui::RESET)
        };
        println!("  {}ROUTER{}     {}", tui::BOLD, tui::RESET, router);

        // Specialist
        let specialist = if trio.specialist_model.is_empty() {
            "\x1b[33m(not set)\x1b[0m".to_string()
        } else {
            format!("{}{}{}", tui::DIM, trio.specialist_model, tui::RESET)
        };
        println!("  {}SPECIALIST{} {}", tui::BOLD, tui::RESET, specialist);

        // Context sizes
        println!(
            "  {}CTX{}        main={}K  router={}K  specialist={}K",
            tui::BOLD,
            tui::RESET,
            self.config.agents.defaults.local_max_context_tokens / 1024,
            trio.router_ctx_tokens / 1024,
            trio.specialist_ctx_tokens / 1024,
        );

        // Loaded models (if LMS managed)
        if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let loaded = crate::lms::list_loaded("", lms_port).await;
            if !loaded.is_empty() {
                println!(
                    "  {}LOADED{}     {}{}{}",
                    tui::BOLD,
                    tui::RESET,
                    tui::DIM,
                    loaded.join(", "),
                    tui::RESET
                );
            }
        }

        // VRAM budget summary (local mode only)
        // migrated from swappable().is_local — phase 09-03
        if self.core_handle.swappable().mode().is_local() {
            let budget = self.compute_current_vram_budget();
            let total_gb = budget.total_vram_bytes as f64 / 1e9;
            let limit_gb = budget.effective_limit_bytes as f64 / 1e9;
            let status = if budget.fits {
                format!("{}OK{}", tui::GREEN, tui::RESET)
            } else {
                format!("\x1b[31mOVER\x1b[0m")
            };
            println!(
                "  {}VRAM{}       {:.1} / {:.1} GB  [{}]",
                tui::BOLD,
                tui::RESET,
                total_gb,
                limit_gb,
                status,
            );
        }

        println!();
    }

    /// Pick a model for a trio role (router or specialist) from LM Studio's available models.
    async fn cmd_trio_pick_model(&mut self, role: &str) {
        // Get available models from LMS
        let models: Vec<String> = if self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            crate::lms::list_available("", lms_port)
                .await
                .into_iter()
                .filter(|m| {
                    // Filter out embedding models
                    !m.to_lowercase().contains("embedding")
                })
                .collect()
        } else {
            Vec::new()
        };

        if models.is_empty() {
            println!("\n  No LLM models found in LM Studio.");
            println!("  Set manually in config.json: trio.{}Model\n", role);
            // Allow manual entry
            let manual_prompt = format!("  Enter {} model ID (or Enter to cancel): ", role);
            let input = match self.rl.as_mut().unwrap().readline(&manual_prompt) {
                Ok(line) => line.trim().to_string(),
                Err(_) => return,
            };
            if input.is_empty() {
                return;
            }
            self.set_trio_role_model(role, &input).await;
            return;
        }

        let current = match role {
            "router" => &self.config.trio.router_model,
            "specialist" => &self.config.trio.specialist_model,
            _ => unreachable!(),
        };

        println!("\n  Available models for {}:", role);
        for (i, model) in models.iter().enumerate() {
            let marker = if crate::lms::is_model_available(std::slice::from_ref(model), current) {
                " (active)"
            } else {
                ""
            };
            println!("  [{}] {}{}", i + 1, model, marker);
        }

        let pick_prompt = format!(
            "  Select {} model [1-{}] or Enter to cancel: ",
            role,
            models.len()
        );
        let choice = match self.rl.as_mut().unwrap().readline(&pick_prompt) {
            Ok(line) => line,
            Err(_) => return,
        };
        let choice = choice.trim();
        if choice.is_empty() {
            return;
        }
        let idx: usize = match choice.parse::<usize>() {
            Ok(n) if n >= 1 && n <= models.len() => n - 1,
            _ => {
                println!("  Invalid selection.\n");
                return;
            }
        };

        let selected = &models[idx];
        self.set_trio_role_model(role, selected).await;
    }

    /// Apply a model selection to a trio role and persist.
    async fn set_trio_role_model(&mut self, role: &str, model: &str) {
        set_trio_role_model_pure(&mut self.config, role, model);

        // Load the model in LMS if trio is active
        if self.config.trio.enabled && self.srv.lms_managed {
            let lms_port = self.config.agents.defaults.lms_port;
            let ctx = match role {
                "router" => Some(self.config.trio.router_ctx_tokens),
                "specialist" => Some(self.config.trio.specialist_ctx_tokens),
                _ => Some(self.config.agents.defaults.local_max_context_tokens),
            };
            print!("  Loading {}... ", model);
            io::stdout().flush().ok();
            match crate::lms::load_model(
                "",
                lms_port,
                model,
                ctx,
                self.config.timeouts.lms_load_secs,
            )
            .await
            {
                Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                Err(e) => println!("{}FAILED: {}{}", tui::RED, e, tui::RESET),
            }
        }

        self.persist_trio_config();
        self.apply_and_rebuild();
        println!(
            "\n  {} {} set to {}{}{}.",
            role.chars()
                .next()
                .unwrap()
                .to_uppercase()
                .collect::<String>()
                + &role[1..],
            tui::BOLD,
            tui::RESET,
            model,
            ""
        );

        // Warn if VRAM budget exceeded after model change
        // migrated from swappable().is_local — phase 09-03
        if self.config.trio.enabled && self.core_handle.swappable().mode().is_local() {
            let budget = self.compute_current_vram_budget();
            if !budget.fits {
                println!(
                    "  \x1b[33mWarning:\x1b[0m VRAM usage ({:.1} GB) exceeds limit ({:.1} GB).",
                    budget.total_vram_bytes as f64 / 1e9,
                    budget.effective_limit_bytes as f64 / 1e9,
                );
                println!("  Use /trio budget for details.");
            }
        }
        println!();
    }

    /// Persist trio + tool_delegation config to disk.
    fn persist_trio_config(&self) {
        let mut disk_cfg = load_config(None);
        persist_trio_fields(&self.config, &mut disk_cfg);
        save_config(&disk_cfg, None);
    }

    pub(super) fn persist_local_config(&self) {
        let mut disk_cfg = load_config(None);
        // Persist local mode settings (including backend so /m switches survive restart)
        disk_cfg.agents.defaults.local_api_base =
            self.config.agents.defaults.local_api_base.clone();
        disk_cfg.agents.defaults.local_backend = self.config.agents.defaults.local_backend.clone();
        disk_cfg.agents.defaults.skip_jit_gate = self.config.agents.defaults.skip_jit_gate;
        disk_cfg.agents.defaults.lms_port = self.config.agents.defaults.lms_port;
        disk_cfg.agents.defaults.higgs_port = self.config.agents.defaults.higgs_port;
        disk_cfg.agents.defaults.lms_main_model =
            self.config.agents.defaults.lms_main_model.clone();
        disk_cfg.agents.defaults.local_model = self.config.agents.defaults.local_model.clone();
        disk_cfg.agents.defaults.mlx_model_dir = self.config.agents.defaults.mlx_model_dir.clone();
        save_config(&disk_cfg, None);
    }

    /// /local — toggle between local and cloud mode.
    pub(super) async fn cmd_local(&mut self) {
        // migrated from swappable().is_local — phase 09-03
        let currently_local = self.core_handle.swappable().mode().is_local();

        if !currently_local {
            // Kill any stale managed inference servers from previous runs
            // before attempting to start a new one (prevents OOM from
            // overlapping servers).
            crate::agent::pid_file::cleanup_stale_pids();

            // Try to start a local inference engine if none is active.
            // For Higgs: always attempt start regardless of local_api_base — it may
            // be stale from a previous session and Higgs may have stopped.
            let is_higgs =
                crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend);
            if (self.config.agents.defaults.local_api_base.is_empty() || is_higgs)
                && !self.srv.lms_managed
                && self.srv.engine == super::super::InferenceEngine::None
            {
                if is_higgs {
                    // Higgs managed sidecar
                    match crate::higgs::resolve_model_dir(&self.config) {
                        Ok(model_dir) => {
                            if let Some(bin) = crate::higgs::find_binary() {
                                let port = self.config.agents.defaults.higgs_port;
                                println!(
                                    "\n  {}{}Higgs{} starting on port {port}...",
                                    tui::BOLD,
                                    tui::YELLOW,
                                    tui::RESET,
                                );
                                match crate::higgs::server_start(
                                    &bin,
                                    port,
                                    &model_dir,
                                    &self.config.agents.defaults.local_model,
                                )
                                .await
                                {
                                    Ok(crate::higgs::StartResult::Ready) => {
                                        self.srv.engine = super::super::InferenceEngine::Higgs;
                                        self.srv.local_port = port.to_string();
                                        self.config.agents.defaults.skip_jit_gate = true;
                                        if let Some(name) = self.apply_higgs_endpoint(port).await {
                                            println!("  {}OK{} · {name}", tui::GREEN, tui::RESET);
                                        } else {
                                            println!("  {}OK{}", tui::GREEN, tui::RESET);
                                        }
                                    }
                                    Ok(crate::higgs::StartResult::Loading {
                                        pid,
                                        port: loading_port,
                                    }) => {
                                        self.srv.engine = super::super::InferenceEngine::Higgs;
                                        self.srv.local_port = loading_port.to_string();
                                        self.config.agents.defaults.skip_jit_gate = true;
                                        self.config.agents.defaults.local_api_base =
                                            format!("http://127.0.0.1:{loading_port}/v1");
                                        println!(
                                            "  {}{}Higgs (pid {}) still loading{} — requests will retry until ready",
                                            tui::BOLD, tui::YELLOW, pid, tui::RESET,
                                        );
                                    }
                                    Err(e) => {
                                        println!(
                                            "  {}{}Higgs start failed:{} {e}",
                                            tui::BOLD,
                                            tui::YELLOW,
                                            tui::RESET,
                                        );
                                        println!(
                                            "  {}Remaining in cloud mode{}\n",
                                            tui::DIM,
                                            tui::RESET
                                        );
                                        return;
                                    }
                                }
                            } else {
                                println!(
                                    "\n  {}{}Higgs not found.{} Install with: cargo install higgs",
                                    tui::BOLD,
                                    tui::YELLOW,
                                    tui::RESET,
                                );
                                return;
                            }
                        }
                        Err(e) => {
                            println!(
                                "\n  {}{}Higgs config error:{} {e}",
                                tui::BOLD,
                                tui::YELLOW,
                                tui::RESET,
                            );
                            return;
                        }
                    }
                } else {
                    // LM Studio path
                    let preference = &self.config.agents.defaults.inference_engine;
                    if let Some((super::super::InferenceEngine::Lms, bin)) =
                        super::super::resolve_inference_engine(preference)
                    {
                        let lms_port = self.config.agents.defaults.lms_port;
                        println!(
                            "\n  {}{}LM Studio{} detected, starting server on port {}...",
                            tui::BOLD,
                            tui::YELLOW,
                            tui::RESET,
                            lms_port
                        );
                        match crate::lms::server_start(&bin, lms_port).await {
                            Ok(()) => {
                                let main_model =
                                    if !self.config.agents.defaults.lms_main_model.is_empty() {
                                        self.config.agents.defaults.lms_main_model.clone()
                                    } else {
                                        let mn = self
                                            .current_model_path
                                            .file_name()
                                            .and_then(|n| n.to_str())
                                            .unwrap_or(&self.config.agents.defaults.local_model);
                                        cli::strip_gguf_suffix(mn).to_string()
                                    };
                                let main_ctx =
                                    Some(self.config.agents.defaults.local_max_context_tokens);
                                print!("  Loading {}... ", main_model);
                                io::stdout().flush().ok();
                                match crate::lms::load_model(
                                    "",
                                    lms_port,
                                    &main_model,
                                    main_ctx,
                                    self.config.timeouts.lms_load_secs,
                                )
                                .await
                                {
                                    Ok(()) => println!("{}OK{}", tui::GREEN, tui::RESET),
                                    Err(e) => {
                                        println!("{}FAILED: {}{}", tui::RED, e, tui::RESET)
                                    }
                                }
                                if self.config.trio.enabled {
                                    if !self.config.trio.router_model.is_empty() {
                                        print!("  Loading {}... ", self.config.trio.router_model);
                                        io::stdout().flush().ok();
                                        match crate::lms::load_model(
                                            "",
                                            lms_port,
                                            &self.config.trio.router_model,
                                            Some(self.config.trio.router_ctx_tokens),
                                            self.config.timeouts.lms_load_secs,
                                        )
                                        .await
                                        {
                                            Ok(()) => {
                                                println!("{}OK{}", tui::GREEN, tui::RESET)
                                            }
                                            Err(e) => {
                                                println!("{}FAILED: {}{}", tui::RED, e, tui::RESET)
                                            }
                                        }
                                    }
                                    if !self.config.trio.specialist_model.is_empty() {
                                        print!(
                                            "  Loading {}... ",
                                            self.config.trio.specialist_model
                                        );
                                        io::stdout().flush().ok();
                                        match crate::lms::load_model(
                                            "",
                                            lms_port,
                                            &self.config.trio.specialist_model,
                                            Some(self.config.trio.specialist_ctx_tokens),
                                            self.config.timeouts.lms_load_secs,
                                        )
                                        .await
                                        {
                                            Ok(()) => {
                                                println!("{}OK{}", tui::GREEN, tui::RESET)
                                            }
                                            Err(e) => {
                                                println!("{}FAILED: {}{}", tui::RED, e, tui::RESET)
                                            }
                                        }
                                    }
                                }
                                self.srv.lms_managed = true;
                                self.srv.lms_binary = Some(bin);
                                self.srv.engine = super::super::InferenceEngine::Lms;
                                self.srv.local_port = lms_port.to_string();
                                if self.config.agents.defaults.local_api_base.is_empty() {
                                    self.config.agents.defaults.local_api_base = format!(
                                        "http://{}:{}/v1",
                                        crate::lms::api_host(),
                                        lms_port
                                    );
                                }
                                self.config.agents.defaults.skip_jit_gate = true;
                            }
                            Err(e) => {
                                println!(
                                    "  {}{}lms server start failed:{} {}",
                                    tui::BOLD,
                                    tui::YELLOW,
                                    tui::RESET,
                                    e
                                );
                                println!("  {}Remaining in cloud mode{}\n", tui::DIM, tui::RESET);
                                return;
                            }
                        }
                    } else {
                        println!(
                            "\n  {}{}No local inference engine found.{} Install LM Studio (lms CLI) or Higgs (cargo install higgs).",
                            tui::BOLD,
                            tui::YELLOW,
                            tui::RESET
                        );
                        return;
                    }
                }
            }

            // When trio strict mode is on but router model is unavailable,
            // disable strict flags so the single model can handle tools directly.
            if self.config.tool_delegation.strict_no_tools_main
                && self.config.tool_delegation.strict_router_schema
            {
                let router_available = if self.srv.lms_managed {
                    let lms_port = self.config.agents.defaults.lms_port;
                    let available = crate::lms::list_available("", lms_port).await;
                    crate::lms::is_model_available(&available, &self.config.trio.router_model)
                } else {
                    !self.config.trio.router_model.is_empty()
                };
                if !router_available {
                    self.config.tool_delegation.strict_no_tools_main = false;
                    self.config.tool_delegation.strict_router_schema = false;
                }
            }

            // Flip to local mode and rebuild.
            self.persist_local_config();
            self.apply_and_rebuild_with(true);
            tui::print_mode_banner(&self.srv.local_port, true);
        } else {
            // Toggle OFF — switch to cloud mode.
            if self.srv.lms_managed {
                let lms_port = self.config.agents.defaults.lms_port;
                crate::lms::unload_all("", lms_port, self.config.timeouts.lms_unload_secs)
                    .await
                    .ok();
                self.srv.lms_managed = false;
                self.srv.lms_binary = None;
            }
            // Clear local endpoint so next /l re-discovers.
            // Preserve "higgs" backend (managed sidecar) so next /l re-starts it.
            // Reset oMLX → lmstudio since oMLX has no auto-start.
            self.config.agents.defaults.local_api_base.clear();
            if !crate::config::schema::is_higgs_backend(&self.config.agents.defaults.local_backend)
            {
                self.config.agents.defaults.local_backend = "lmstudio".to_string();
            }
            self.config.agents.defaults.skip_jit_gate = false;
            self.srv.engine = super::super::InferenceEngine::None;
            self.stop_watchdog();
            self.persist_local_config();
            self.apply_and_rebuild_with(false);
            tui::print_mode_banner(&self.srv.local_port, false);
        }
    }

    /// /lane [answer|action] — switch execution lane or toggle.
    pub(super) fn cmd_lane(&mut self, arg: &str) {
        use crate::agent::lane::Lane;

        let current = self.core_handle.swappable().lane;
        let new_lane = match arg.trim().to_lowercase().as_str() {
            "answer" => Lane::Answer,
            "action" => Lane::Action,
            "" => {
                // Toggle
                match current {
                    Lane::Answer => Lane::Action,
                    Lane::Action => Lane::Answer,
                }
            }
            other => {
                println!("\n  Unknown lane: {}. Use 'answer' or 'action'.\n", other);
                return;
            }
        };
        // Persist to config so lane survives core rebuild (model switch).
        self.config.agents.default_lane = Some(new_lane.to_string());
        self.apply_and_rebuild();
        println!("\n  Lane switched to: {}\n", new_lane);
    }
}
