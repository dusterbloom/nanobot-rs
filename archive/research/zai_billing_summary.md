# Z.AI Billing Structure - Quick Reference

## Question 1: Subscription Plan or Pay-Per-Token?
**Answer: BOTH (Two Separate Systems)**

| Aspect | GLM Coding Plan | Standard API |
|--------|-----------------|--------------|
| **Type** | Subscription | Pay-per-token |
| **Price** | $3-6/month (Lite), $15-30/month (Pro), higher (Max) | $0.06-$0.75 per 1M input tokens |
| **Usage** | Coding tools only (Claude Code, Cline, etc.) | General API access |
| **Quota** | Tens of billions tokens/month | Per-token billing |
| **Endpoint** | `api.z.ai/api/anthropic` or `api.z.ai/api/coding/paas/v4` | `api.z.ai/api/paas/v4/chat/completions` |
| **Billing Isolation** | Separate quota system | Separate from Coding Plan |

**Critical Point**: API calls are billed separately and do NOT use Coding Plan quota.

---

## Question 2: Unlimited API Plan?
**Answer: NO**

- No unlimited API plan exists
- Standard API is always pay-per-token
- GLM Coding Plans provide massive allowances (~1% of standard API pricing) but ONLY for coding tools
- Users report 40-60M tokens/day without hitting limits on $3 plans, but this is for coding tools only

---

## Question 3: Web Chat Subscription vs API Billing?
**Answer: No Separate Web Chat Subscription**

- **Web Chat (chat.z.ai)**: FREE (no paid tier)
- **API Access**: Pay-per-token (separate billing)
- **Coding Tools**: GLM Coding Plan subscription (separate system)

Z.ai does NOT have a ChatGPT Plus equivalent for web chat. The free chat interface and paid options are completely separate products.

---

## Summary
Z.ai uses a **hybrid three-tier model**:
1. **Free Web Chat** - No subscription needed
2. **Subscription Coding Plans** - For IDE/coding tool integration (~$3-6/month)
3. **Pay-Per-Token API** - For direct API access (starts at $0.06 per 1M input tokens)

Each tier has independent billing and does not interact with the others.
