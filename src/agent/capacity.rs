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
    pub(crate) schema_version: u32,
    pub(crate) model: String,
    pub(crate) model_fingerprint: String,
    pub(crate) boot_id: String,
    pub(crate) generation: u64,
    pub(crate) availability: CapacityAvailability,
    pub(crate) pressure: CapacityPressure,
    pub(crate) safe_total_tokens: u64,
    pub(crate) recommended_output_tokens: u64,
    pub(crate) max_prompt_tokens: u64,
    pub(crate) retained_session_tokens: u64,
    pub(crate) retained_bytes: u64,
    pub(crate) prefix_cache_bytes: u64,
    pub(crate) basis: CapacityBasis,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EffectiveCapacity {
    pub(crate) total_tokens: usize,
    pub(crate) max_prompt_tokens: usize,
    pub(crate) output_tokens: usize,
}

impl EffectiveCapacity {
    pub(crate) fn legacy_higgs(configured: &TokenBudget, immutable_prefix_tokens: usize) -> Self {
        effective_capacity_from_limits(16_384, 4_096, 12_288, configured, immutable_prefix_tokens)
    }

    pub(crate) fn prompt_room(
        &self,
        planned_output: usize,
        protocol_overhead: usize,
    ) -> Result<usize, CapacityError> {
        let reserved = planned_output
            .checked_add(protocol_overhead)
            .ok_or(CapacityError::Overflow)?;
        let total_room = self
            .total_tokens
            .checked_sub(reserved)
            .ok_or(CapacityError::Unavailable)?;
        Ok(self.max_prompt_tokens.min(total_room))
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
        Ok(effective_capacity_from_limits(
            safe_total,
            recommended_output,
            server_prompt,
            configured,
            immutable_prefix_tokens,
        ))
    }
}

fn effective_capacity_from_limits(
    safe_total: usize,
    recommended_output: usize,
    server_prompt: usize,
    configured: &TokenBudget,
    immutable_prefix_tokens: usize,
) -> EffectiveCapacity {
    let total_tokens = safe_total.min(configured.max_context());
    let output_tokens = recommended_output
        .min(configured.response_reserve())
        .min(total_tokens.saturating_sub(immutable_prefix_tokens));
    let max_prompt_tokens = server_prompt.min(total_tokens.saturating_sub(output_tokens));
    EffectiveCapacity {
        total_tokens,
        max_prompt_tokens,
        output_tokens,
    }
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
        assert_eq!(effective.prompt_room(4_096, 512), Ok(48_640));
        assert_eq!(
            effective.prompt_room(usize::MAX, 1),
            Err(CapacityError::Overflow)
        );
    }

    #[test]
    fn legacy_higgs_fallback_is_bounded_to_16k_total_and_4k_output() {
        assert_eq!(
            EffectiveCapacity::legacy_higgs(&TokenBudget::new(100_000, 8_000), 1_000),
            EffectiveCapacity {
                total_tokens: 16_384,
                max_prompt_tokens: 12_288,
                output_tokens: 4_096,
            }
        );
        assert_eq!(
            EffectiveCapacity::legacy_higgs(&TokenBudget::new(10_000, 2_000), 1_000),
            EffectiveCapacity {
                total_tokens: 10_000,
                max_prompt_tokens: 8_000,
                output_tokens: 2_000,
            }
        );
    }
}
