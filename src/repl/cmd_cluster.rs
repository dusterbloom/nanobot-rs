//! Cluster REPL commands: /cluster, /skill.

use super::*;

impl ReplContext {
    /// /cluster — show cluster peers, models, and routing status.
    #[cfg(feature = "cluster")]
    pub(super) async fn cmd_cluster(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        match parts.as_slice() {
            [] | ["status"] => self.cmd_cluster_status().await,
            ["models"] => self.cmd_cluster_models().await,
            ["peers"] => self.cmd_cluster_peers().await,
            ["use", rest @ ..] => self.cmd_cluster_use(&rest.join(" ")).await,
            _ => {
                println!(
                    "
  Usage: /cluster [subcommand]"
                );
                println!("    /cluster              Show cluster status");
                println!("    /cluster peers        List discovered peers");
                println!("    /cluster models       List all models across peers");
                println!(
                    "    /cluster use <model>  Switch to a model on a cluster peer
"
                );
            }
        }
    }

    #[cfg(not(feature = "cluster"))]
    pub(super) async fn cmd_cluster(&mut self, _arg: &str) {
        println!(
            "
  Cluster mode not available (compile with --features cluster).
"
        );
    }

    #[cfg(feature = "cluster")]
    async fn cmd_cluster_status(&self) {
        let state = match &self.cluster_state {
            Some(s) => s,
            None => {
                println!(
                    "
  Cluster not enabled. Add to config:"
                );
                println!("    \"cluster\": {{ \"enabled\": true }}\n");
                return;
            }
        };

        let peers = state.get_all_peers().await;
        let healthy = peers.iter().filter(|p| p.healthy).count();
        let total_models: usize = peers.iter().map(|p| p.models.len()).sum();

        println!(
            "
  {}CLUSTER{}",
            tui::BOLD,
            tui::RESET
        );
        println!("    Peers: {} ({} healthy)", peers.len(), healthy);
        println!("    Models: {} total", total_models);

        for peer in &peers {
            let status = if peer.healthy { "●" } else { "○" };
            let age = peer.last_seen.elapsed().as_secs();
            println!(
                "    {} {} ({}) — {} models, seen {}s ago",
                status,
                peer.endpoint,
                peer.peer_type,
                peer.models.len(),
                age
            );
        }
        println!();
    }

    #[cfg(feature = "cluster")]
    async fn cmd_cluster_peers(&self) {
        let state = match &self.cluster_state {
            Some(s) => s,
            None => {
                println!(
                    "
  Cluster not enabled.
"
                );
                return;
            }
        };

        let peers = state.get_all_peers().await;
        if peers.is_empty() {
            println!(
                "
  No peers discovered.
"
            );
            return;
        }

        println!(
            "
  Discovered peers:"
        );
        for (i, peer) in peers.iter().enumerate() {
            let status = if peer.healthy { "healthy" } else { "down" };
            println!(
                "    {}. {} [{}] ({})",
                i + 1,
                peer.endpoint,
                peer.peer_type,
                status
            );
            println!("       Models: {}", peer.models.len());
            if let Some(vram) = peer.total_vram_mb {
                println!("       VRAM: {} MB", vram);
            }
        }
        println!();
    }

    #[cfg(feature = "cluster")]
    async fn cmd_cluster_models(&self) {
        let state = match &self.cluster_state {
            Some(s) => s,
            None => {
                println!(
                    "
  Cluster not enabled.
"
                );
                return;
            }
        };

        let peers = state.get_healthy_peers().await;
        if peers.is_empty() {
            println!(
                "
  No healthy peers found.
"
            );
            return;
        }

        println!(
            "
  Cluster models:"
        );
        for peer in &peers {
            // Extract short name from endpoint (e.g. "192.168.1.62" from "http://192.168.1.62:1234/v1")
            let short = peer
                .endpoint
                .trim_start_matches("http://")
                .trim_start_matches("https://")
                .split(':')
                .next()
                .unwrap_or(&peer.endpoint);

            println!(
                "
    {} ({}):",
                short, peer.peer_type
            );
            for model in &peer.models {
                println!("      - {}", model.id);
            }
        }
        println!();
    }

    /// `/cluster use <model>` — switch the active model to one on a cluster peer.
    #[cfg(feature = "cluster")]
    async fn cmd_cluster_use(&mut self, query: &str) {
        if query.is_empty() {
            println!("\n  Usage: /cluster use <model>\n  Example: /cluster use qwen3.5-35b\n");
            return;
        }

        let state = match &self.cluster_state {
            Some(s) => s.clone(),
            None => {
                println!("\n  Cluster not enabled.\n");
                return;
            }
        };

        let found = state.find_model(query).await;
        match found {
            Some(m) => {
                let peer_endpoint = &m.peer.endpoint;
                let model_id = &m.model.id;

                println!(
                    "\n  Switching to {}{}{}  on  {}",
                    tui::BOLD,
                    model_id,
                    tui::RESET,
                    peer_endpoint,
                );

                // Update in-memory config.
                self.config.agents.defaults.local_api_base = peer_endpoint.clone();
                self.config.agents.defaults.lms_main_model = model_id.clone();
                self.config.agents.defaults.local_model = model_id.clone();
                // Switch backend away from "mlx" (in-process) so rebuild uses
                // the HTTP provider for this cluster peer endpoint.
                self.config.agents.defaults.local_backend = "omlx".to_string();
                self.srv.lms_managed = false;
                self.current_model_path = PathBuf::from(model_id);

                // Persist to disk.
                self.persist_local_config();

                // Rebuild the agent core with the new model/endpoint.
                self.apply_and_rebuild_with(true);

                println!("  {}Done.{}\n", tui::GREEN, tui::RESET);
            }
            None => {
                println!("\n  No healthy peer has a model matching \"{}\".", query);
                println!("  Use /cluster models to see available models.\n");
            }
        }
    }

    /// /skill — manage skills (add, remove, list).
    pub(super) async fn cmd_skill(&mut self, arg: &str) {
        let parts: Vec<&str> = arg.split_whitespace().collect();
        let workspace = self.core_handle.swappable().workspace.clone();

        match parts.as_slice() {
            [] | ["list"] => {
                let loader = crate::agent::skills::SkillsLoader::new(&workspace, None);
                let skills = loader.list_skills(false);
                if skills.is_empty() {
                    println!("\n  No skills installed.\n");
                } else {
                    println!("\n  {} skill(s):\n", skills.len());
                    for s in &skills {
                        let desc = loader
                            .get_skill_metadata(&s.name)
                            .and_then(|m| m.get("description").cloned())
                            .unwrap_or_else(|| "(no description)".to_string());
                        let version = loader
                            .get_skill_metadata(&s.name)
                            .and_then(|m| m.get("version").cloned())
                            .map(|v| format!(" v{}", v))
                            .unwrap_or_default();
                        println!("    [{}]{} {} — {}", s.source, version, s.name, desc);
                    }
                    println!();
                }
            }
            ["add", source] => {
                println!("\n  Installing from {}...", source);
                match cli::cmd_skill_add(&workspace, source).await {
                    Ok(installed) => {
                        for name in &installed {
                            println!("    Installed: {}", name);
                        }
                        println!("  {} skill(s) installed.\n", installed.len());
                    }
                    Err(e) => println!("  Error: {}\n", e),
                }
            }
            ["remove", name] | ["rm", name] => match cli::cmd_skill_remove(&workspace, name) {
                Ok(()) => println!("\n  Removed skill: {}\n", name),
                Err(e) => println!("\n  Error: {}\n", e),
            },
            _ => {
                println!(
                    "
  Usage: /skill [subcommand]
    /skill              List installed skills
    /skill list         List installed skills
    /skill add <src>    Install from GitHub (owner/repo or owner/repo@skill)
    /skill remove <n>   Remove a skill by name
"
                );
            }
        }
    }
}
