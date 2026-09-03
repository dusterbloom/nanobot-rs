use serde::{Deserialize, Serialize};

use crate::agent::token_budget::TokenBudget;

const CAPACITY_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapacityAvailability {
    Available,
    Unavailable,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapacityPressure {
    Normal,
    Constrained,
    Critical,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapacityBasis {
    Conservative,
    Learned,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawHiggsCapacityProfile {
    schema_version: u32,
    model: String,
    model_fingerprint: String,
    boot_id: String,
    generation: u64,
    availability: CapacityAvailability,
    pressure: CapacityPressure,
    safe_total_tokens: u64,
    recommended_output_tokens: u64,
    max_prompt_tokens: u64,
    retained_session_tokens: u64,
    retained_bytes: u64,
    prefix_cache_bytes: u64,
    basis: CapacityBasis,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase", try_from = "RawHiggsCapacityProfile")]
pub(crate) struct HiggsCapacityProfile {
    schema_version: u32,
    model: String,
    model_fingerprint: String,
    boot_id: String,
    generation: u64,
    availability: CapacityAvailability,
    pressure: CapacityPressure,
    safe_total_tokens: u64,
    recommended_output_tokens: u64,
    max_prompt_tokens: u64,
    retained_session_tokens: u64,
    retained_bytes: u64,
    prefix_cache_bytes: u64,
    basis: CapacityBasis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EffectiveCapacity {
    /// Tokens occupied by the immutable system/tool prefix inside the prompt cap.
    immutable_prefix_tokens: usize,
    /// Whole request envelope, including prompt and generated output.
    pub(crate) total_tokens: usize,
    /// Whole prompt envelope, including immutable prefix and protocol overhead.
    pub(crate) max_prompt_tokens: usize,
    pub(crate) output_tokens: usize,
}

/// One `/v1/capacity` fetch result: the live profile, or the marker for an
/// old Higgs whose route-absent 404 selects the conservative legacy fallback.
#[derive(Debug)]
pub(crate) enum HiggsCapacityFetch {
    Profile(HiggsCapacityProfile),
    Legacy,
}

impl EffectiveCapacity {
    pub(crate) fn legacy_higgs(
        configured: &TokenBudget,
        immutable_prefix_tokens: usize,
    ) -> Result<Self, CapacityError> {
        effective_capacity_from_limits(16_384, 4_096, 12_288, configured, immutable_prefix_tokens)
    }

    pub(crate) fn prompt_room(
        &self,
        planned_output: usize,
        protocol_overhead: usize,
    ) -> Result<usize, CapacityError> {
        let prompt_reserved = self
            .immutable_prefix_tokens
            .checked_add(protocol_overhead)
            .ok_or(CapacityError::Overflow)?;
        let prompt_room = self
            .max_prompt_tokens
            .checked_sub(prompt_reserved)
            .ok_or(CapacityError::Unavailable)?;
        let request_reserved = prompt_reserved
            .checked_add(planned_output)
            .ok_or(CapacityError::Overflow)?;
        let request_room = self
            .total_tokens
            .checked_sub(request_reserved)
            .ok_or(CapacityError::Unavailable)?;
        Ok(prompt_room.min(request_room))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub(crate) enum CapacityError {
    #[error("Higgs capacity is unavailable")]
    Unavailable,
    #[error("Higgs capacity does not fit this platform's token size")]
    Overflow,
}

impl HiggsCapacityProfile {
    pub(crate) fn schema_version(&self) -> u32 {
        self.schema_version
    }

    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn model_fingerprint(&self) -> &str {
        &self.model_fingerprint
    }

    pub(crate) fn boot_id(&self) -> &str {
        &self.boot_id
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    pub(crate) fn availability(&self) -> CapacityAvailability {
        self.availability
    }

    pub(crate) fn pressure(&self) -> CapacityPressure {
        self.pressure
    }

    pub(crate) fn basis(&self) -> CapacityBasis {
        self.basis
    }

    /// Capacity generations are comparable only within one Higgs process boot.
    pub(crate) fn is_same_revision(&self, other: &Self) -> bool {
        self.boot_id == other.boot_id && self.generation == other.generation
    }

    pub(crate) fn effective_capacity(
        &self,
        configured: &TokenBudget,
        immutable_prefix_tokens: usize,
    ) -> Result<EffectiveCapacity, CapacityError> {
        if self.availability == CapacityAvailability::Unavailable {
            return Err(CapacityError::Unavailable);
        }

        let safe_total =
            usize::try_from(self.safe_total_tokens).map_err(|_| CapacityError::Overflow)?;
        let recommended_output =
            usize::try_from(self.recommended_output_tokens).map_err(|_| CapacityError::Overflow)?;
        let server_prompt =
            usize::try_from(self.max_prompt_tokens).map_err(|_| CapacityError::Overflow)?;
        effective_capacity_from_limits(
            safe_total,
            recommended_output,
            server_prompt,
            configured,
            immutable_prefix_tokens,
        )
    }
}

/// Loop-level holder for the live Higgs capacity snapshot. Lives beside
/// `RuntimeCounters` on the `AgentHandle` (persists across core swaps), keyed
/// by endpoint+model so an unchanged provider tuple never refetches. The
/// configured `SwappableCore.token_budget` stays immutable; the effective
/// request budget is derived at use time and can shrink per boot/generation
/// without a config rewrite or core rebuild.
pub(crate) struct CapacityRuntime {
    state: std::sync::Mutex<CapacityRuntimeState>,
}

#[derive(Default)]
struct CapacityRuntimeState {
    /// Identity of the installed snapshot: endpoint + model.
    key: Option<(String, String)>,
    /// The installed live snapshot, or the frozen legacy fallback.
    installed: Option<InstalledCapacity>,
}

enum InstalledCapacity {
    Profile(Box<HiggsCapacityProfile>),
    Legacy,
}

/// What a refresh observed, so the caller can rotate retained state through
/// the existing epoch path when the server restarted underneath it.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum CapacityRefresh {
    /// No snapshot installed yet, or the endpoint/model changed.
    Fetched,
    /// Same endpoint+model tuple as the installed snapshot: no fetch.
    Unchanged,
    /// Same endpoint+model but the boot ID changed: the snapshot was
    /// re-fetched and every retained session from the old boot is stale.
    BootChanged,
    /// Endpoint+model changed away from a previously installed Higgs pair
    /// (model switch or cloud takeover): the snapshot is dropped.
    Invalidated,
}

impl CapacityRuntime {
    pub(crate) fn shared() -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self::default())
    }

    /// Whether a live snapshot is already installed for this endpoint+model.
    /// The loop consults this before issuing a fetch: an unchanged tuple
    /// never refetches.
    pub(crate) fn cached_for(&self, endpoint: &str, model: &str) -> bool {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        matches!(&state.key, Some(key) if key.0 == endpoint && key.1 == model)
    }

    /// Install a fetched profile. Returns the refresh classification for the
    /// caller's retained-epoch handling.
    pub(crate) fn install_profile(
        &self,
        endpoint: &str,
        model: &str,
        profile: HiggsCapacityProfile,
    ) -> CapacityRefresh {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let refresh = match (&state.key, &state.installed) {
            (Some(key), Some(InstalledCapacity::Profile(previous))) if key.0 == endpoint => {
                if previous.boot_id() == profile.boot_id() {
                    CapacityRefresh::Unchanged
                } else {
                    CapacityRefresh::BootChanged
                }
            }
            _ => CapacityRefresh::Fetched,
        };
        state.key = Some((endpoint.to_owned(), model.to_owned()));
        state.installed = Some(InstalledCapacity::Profile(Box::new(profile)));
        refresh
    }

    /// Freeze the conservative legacy fallback for an old Higgs without
    /// `/v1/capacity`: 16,384 total tokens, at most 4,096 output.
    pub(crate) fn install_legacy(&self, endpoint: &str, model: &str) -> CapacityRefresh {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let refresh = if state.installed.is_some() {
            CapacityRefresh::BootChanged
        } else {
            CapacityRefresh::Fetched
        };
        state.key = Some((endpoint.to_owned(), model.to_owned()));
        state.installed = Some(InstalledCapacity::Legacy);
        refresh
    }

    /// Drop the installed snapshot (model switch, runtime toggle). The next
    /// Higgs-capable request refetches from discovery.
    pub(crate) fn invalidate(&self) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.installed.is_some() {
            state.key = None;
            state.installed = None;
        }
    }

    /// The effective request budget: the configured ceiling narrowed by the
    /// live snapshot. Without a snapshot (cloud provider, pre-discovery,
    /// unavailable profile) this is exactly the configured budget.
    pub(crate) fn effective_budget(
        &self,
        configured: &TokenBudget,
        immutable_prefix_tokens: usize,
    ) -> TokenBudget {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match &state.installed {
            Some(InstalledCapacity::Profile(profile)) => {
                match profile.effective_capacity(configured, immutable_prefix_tokens) {
                    Ok(effective) => TokenBudget::new(
                        effective.total_tokens.min(configured.max_context()),
                        effective.output_tokens.min(configured.response_reserve()),
                    ),
                    // Unavailable profile: keep the configured ceiling; the
                    // request path surfaces typed 503 from the server rather
                    // than silently shrinking to zero.
                    Err(_) => {
                        TokenBudget::new(configured.max_context(), configured.response_reserve())
                    }
                }
            }
            Some(InstalledCapacity::Legacy) => {
                match EffectiveCapacity::legacy_higgs(configured, immutable_prefix_tokens) {
                    Ok(effective) => {
                        TokenBudget::new(effective.total_tokens, effective.output_tokens)
                    }
                    Err(_) => {
                        TokenBudget::new(configured.max_context(), configured.response_reserve())
                    }
                }
            }
            None => TokenBudget::new(configured.max_context(), configured.response_reserve()),
        }
    }
}

impl Default for CapacityRuntime {
    fn default() -> Self {
        Self {
            state: std::sync::Mutex::new(CapacityRuntimeState::default()),
        }
    }
}

fn effective_capacity_from_limits(
    safe_total: usize,
    recommended_output: usize,
    server_prompt: usize,
    configured: &TokenBudget,
    immutable_prefix_tokens: usize,
) -> Result<EffectiveCapacity, CapacityError> {
    let total_tokens = safe_total.min(configured.max_context());
    let prefix_room = total_tokens
        .checked_sub(immutable_prefix_tokens)
        .ok_or(CapacityError::Unavailable)?;
    let output_tokens = recommended_output
        .min(configured.response_reserve())
        .min(prefix_room);
    let request_prompt = total_tokens
        .checked_sub(output_tokens)
        .ok_or(CapacityError::Overflow)?;
    let max_prompt_tokens = server_prompt.min(request_prompt);
    Ok(EffectiveCapacity {
        immutable_prefix_tokens,
        total_tokens,
        max_prompt_tokens,
        output_tokens,
    })
}

impl TryFrom<RawHiggsCapacityProfile> for HiggsCapacityProfile {
    type Error = String;

    fn try_from(raw: RawHiggsCapacityProfile) -> Result<Self, Self::Error> {
        if raw.schema_version != CAPACITY_SCHEMA_VERSION {
            return Err(format!(
                "unsupported capacity schemaVersion {}; expected {CAPACITY_SCHEMA_VERSION}",
                raw.schema_version
            ));
        }
        if raw.model.trim().is_empty() {
            return Err("capacity model must not be empty".to_owned());
        }
        if raw.model_fingerprint.trim().is_empty() {
            return Err("capacity modelFingerprint must not be empty".to_owned());
        }
        if raw.boot_id.trim().is_empty() {
            return Err("capacity bootId must not be empty".to_owned());
        }
        if raw.availability == CapacityAvailability::Available {
            if raw.safe_total_tokens == 0
                || raw.recommended_output_tokens == 0
                || raw.max_prompt_tokens == 0
                || raw.recommended_output_tokens > raw.safe_total_tokens
                || raw.max_prompt_tokens > raw.safe_total_tokens - raw.recommended_output_tokens
                || raw.retained_session_tokens > raw.max_prompt_tokens
            {
                return Err("invalid available capacity token relationships".to_owned());
            }
        } else if raw.safe_total_tokens != 0
            || raw.recommended_output_tokens != 0
            || raw.max_prompt_tokens != 0
            || raw.retained_session_tokens != 0
            || raw.retained_bytes != 0
            || raw.prefix_cache_bytes != 0
        {
            return Err("unavailable capacity limits must all be zero".to_owned());
        }
        Ok(Self {
            schema_version: raw.schema_version,
            model: raw.model,
            model_fingerprint: raw.model_fingerprint,
            boot_id: raw.boot_id,
            generation: raw.generation,
            availability: raw.availability,
            pressure: raw.pressure,
            safe_total_tokens: raw.safe_total_tokens,
            recommended_output_tokens: raw.recommended_output_tokens,
            max_prompt_tokens: raw.max_prompt_tokens,
            retained_session_tokens: raw.retained_session_tokens,
            retained_bytes: raw.retained_bytes,
            prefix_cache_bytes: raw.prefix_cache_bytes,
            basis: raw.basis,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::token_budget::TokenBudget;
    use serde_json::json;

    fn available_profile() -> serde_json::Value {
        json!({
            "schemaVersion": 1,
            "model": "escha-35b-a3b",
            "modelFingerprint": "sha256:abc",
            "bootId": "boot-1",
            "generation": 7,
            "availability": "available",
            "pressure": "normal",
            "safeTotalTokens": 53_248,
            "recommendedOutputTokens": 4_096,
            "maxPromptTokens": 49_152,
            "retainedSessionTokens": 49_152,
            "retainedBytes": 2_147_483_648_u64,
            "prefixCacheBytes": 1_073_741_824_u64,
            "basis": "learned"
        })
    }

    #[test]
    fn rejects_non_v1_schema() {
        let mut value = available_profile();
        value["schemaVersion"] = json!(2);

        assert!(serde_json::from_value::<HiggsCapacityProfile>(value).is_err());
    }

    #[test]
    fn capacity_profile_round_trips_exact_camel_case_schema() {
        let value = available_profile();
        let profile = serde_json::from_value::<HiggsCapacityProfile>(value.clone()).unwrap();

        assert_eq!(serde_json::to_value(profile).unwrap(), value);
    }

    #[test]
    fn exposes_narrow_validated_identity_and_status_views() {
        let profile = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();

        assert_eq!(profile.schema_version(), 1);
        assert_eq!(profile.model(), "escha-35b-a3b");
        assert_eq!(profile.model_fingerprint(), "sha256:abc");
        assert_eq!(profile.boot_id(), "boot-1");
        assert_eq!(profile.generation(), 7);
        assert_eq!(profile.availability(), CapacityAvailability::Available);
        assert_eq!(profile.pressure(), CapacityPressure::Normal);
        assert_eq!(profile.basis(), CapacityBasis::Learned);
    }

    #[test]
    fn rejects_empty_fingerprint_or_boot_id() {
        for field in ["modelFingerprint", "bootId"] {
            let mut value = available_profile();
            value[field] = json!("");
            assert!(
                serde_json::from_value::<HiggsCapacityProfile>(value).is_err(),
                "{field} must identify the capacity source"
            );
        }
    }

    #[test]
    fn rejects_blank_model_name() {
        let mut value = available_profile();
        value["model"] = json!(" \t\n");

        assert!(serde_json::from_value::<HiggsCapacityProfile>(value).is_err());
    }

    #[test]
    fn rejects_unknown_enum_values_and_fields() {
        for (field, value) in [
            ("availability", json!("loading")),
            ("pressure", json!("warning")),
            ("basis", json!("measured")),
        ] {
            let mut profile = available_profile();
            profile[field] = value;
            assert!(serde_json::from_value::<HiggsCapacityProfile>(profile).is_err());
        }

        let mut profile = available_profile();
        profile["unsafeExtraLimit"] = json!(999_999);
        assert!(serde_json::from_value::<HiggsCapacityProfile>(profile).is_err());
    }

    #[test]
    fn rejects_invalid_available_field_relationships() {
        let invalid = [
            ("safeTotalTokens", json!(0)),
            ("recommendedOutputTokens", json!(0)),
            ("maxPromptTokens", json!(0)),
            ("maxPromptTokens", json!(50_000)),
            ("retainedSessionTokens", json!(49_153)),
        ];

        for (field, replacement) in invalid {
            let mut value = available_profile();
            value[field] = replacement;
            assert!(
                serde_json::from_value::<HiggsCapacityProfile>(value).is_err(),
                "invalid {field} relationship must be rejected"
            );
        }
    }

    #[test]
    fn unavailable_profile_has_no_effective_budget() {
        let mut value = available_profile();
        value["availability"] = json!("unavailable");
        for field in [
            "safeTotalTokens",
            "recommendedOutputTokens",
            "maxPromptTokens",
            "retainedSessionTokens",
            "retainedBytes",
            "prefixCacheBytes",
        ] {
            value[field] = json!(0);
        }
        let profile = serde_json::from_value::<HiggsCapacityProfile>(value).unwrap();

        assert_eq!(
            profile.effective_capacity(&TokenBudget::new(64_000, 8_000), 2_000),
            Err(CapacityError::Unavailable)
        );
    }

    #[test]
    fn rejects_unavailable_profile_with_nonzero_limits() {
        for field in [
            "safeTotalTokens",
            "recommendedOutputTokens",
            "maxPromptTokens",
            "retainedSessionTokens",
            "retainedBytes",
            "prefixCacheBytes",
        ] {
            let mut value = available_profile();
            value["availability"] = json!("unavailable");
            for zero_field in [
                "safeTotalTokens",
                "recommendedOutputTokens",
                "maxPromptTokens",
                "retainedSessionTokens",
                "retainedBytes",
                "prefixCacheBytes",
            ] {
                value[zero_field] = json!(0);
            }
            value[field] = json!(1);
            assert!(
                serde_json::from_value::<HiggsCapacityProfile>(value).is_err(),
                "unavailable {field} must remain zero"
            );
        }
    }

    #[test]
    fn effective_budget_applies_server_and_config_ceiling_minima() {
        let profile = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();

        assert_eq!(
            profile
                .effective_capacity(&TokenBudget::new(100_000, 8_000), 2_000)
                .unwrap(),
            EffectiveCapacity {
                immutable_prefix_tokens: 2_000,
                total_tokens: 53_248,
                max_prompt_tokens: 49_152,
                output_tokens: 4_096,
            }
        );
        assert_eq!(
            profile
                .effective_capacity(&TokenBudget::new(32_000, 3_000), 2_000)
                .unwrap(),
            EffectiveCapacity {
                immutable_prefix_tokens: 2_000,
                total_tokens: 32_000,
                max_prompt_tokens: 29_000,
                output_tokens: 3_000,
            }
        );
    }

    #[test]
    fn generation_identity_includes_boot_id() {
        let current = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();
        let same = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();
        let mut restarted_value = available_profile();
        restarted_value["bootId"] = json!("boot-2");
        let restarted = serde_json::from_value::<HiggsCapacityProfile>(restarted_value).unwrap();

        assert!(current.is_same_revision(&same));
        assert!(!current.is_same_revision(&restarted));
    }

    #[test]
    fn reserves_prefix_output_and_protocol_room_with_checked_arithmetic() {
        let profile = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();
        let constrained = profile
            .effective_capacity(&TokenBudget::new(100_000, 8_000), 52_000)
            .unwrap();
        assert_eq!(constrained.output_tokens, 1_248);

        let effective = profile
            .effective_capacity(&TokenBudget::new(100_000, 8_000), 2_000)
            .unwrap();
        assert_eq!(effective.prompt_room(4_096, 512), Ok(46_640));
        assert_eq!(
            effective.prompt_room(usize::MAX, 1),
            Err(CapacityError::Overflow)
        );
    }

    #[test]
    fn whole_prompt_cap_also_reserves_prefix_and_protocol_overhead() {
        let mut value = available_profile();
        value["maxPromptTokens"] = json!(40_000);
        value["retainedSessionTokens"] = json!(40_000);
        let profile = serde_json::from_value::<HiggsCapacityProfile>(value).unwrap();
        let effective = profile
            .effective_capacity(&TokenBudget::new(100_000, 8_000), 2_000)
            .unwrap();

        assert_eq!(effective.max_prompt_tokens, 40_000);
        assert_eq!(effective.prompt_room(4_096, 500), Ok(37_500));
    }

    #[test]
    fn legacy_higgs_fallback_is_bounded_to_16k_total_and_4k_output() {
        let fallback =
            EffectiveCapacity::legacy_higgs(&TokenBudget::new(100_000, 8_000), 1_000).unwrap();
        assert_eq!(
            fallback,
            EffectiveCapacity {
                immutable_prefix_tokens: 1_000,
                total_tokens: 16_384,
                max_prompt_tokens: 12_288,
                output_tokens: 4_096,
            }
        );
        assert_eq!(fallback.prompt_room(4_096, 288), Ok(11_000));
        assert_eq!(
            EffectiveCapacity::legacy_higgs(&TokenBudget::new(10_000, 2_000), 1_000).unwrap(),
            EffectiveCapacity {
                immutable_prefix_tokens: 1_000,
                total_tokens: 10_000,
                max_prompt_tokens: 8_000,
                output_tokens: 2_000,
            }
        );
    }

    #[test]
    fn rejects_immutable_prefix_larger_than_effective_total() {
        let profile = serde_json::from_value::<HiggsCapacityProfile>(available_profile()).unwrap();
        assert_eq!(
            profile.effective_capacity(&TokenBudget::new(1_000, 100), 1_001),
            Err(CapacityError::Unavailable)
        );
        assert_eq!(
            EffectiveCapacity::legacy_higgs(&TokenBudget::new(1_000, 100), 1_001),
            Err(CapacityError::Unavailable)
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn rejects_wire_token_counts_that_overflow_usize() {
        let mut value = available_profile();
        value["safeTotalTokens"] = json!(u64::from(u32::MAX) + 2);
        value["recommendedOutputTokens"] = json!(1);
        value["maxPromptTokens"] = json!(u64::from(u32::MAX));
        value["retainedSessionTokens"] = json!(0);
        let profile = serde_json::from_value::<HiggsCapacityProfile>(value).unwrap();

        assert_eq!(
            profile.effective_capacity(&TokenBudget::new(usize::MAX, 1), 0),
            Err(CapacityError::Overflow)
        );
    }
}
