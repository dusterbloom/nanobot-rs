# Z.AI (Zhipu AI / BigModel) API Billing Research

## Research Summary
Comprehensive analysis of Z.ai's billing structure for API usage, including subscription plans and pay-per-token options.

---

## Key Findings

### 1. Does Z.ai offer a subscription plan that covers API calls? Or is it strictly pay-per-token?

**Answer: BOTH - Two Separate Systems**

Z.ai offers **two distinct billing systems** that operate independently:

#### A. GLM Coding Plan (Subscription-Based)
- **Purpose**: Subscription specifically for AI coding tools (Claude Code, Cline, OpenCode, Kilo Code, Roo Code, etc.)
- **Pricing Tiers**:
  - **Lite Plan**: Starting at $3-6/month (promotional vs. regular)
    - ~80 prompts per 5-hour cycle
    - ~3× the usage quota of Claude Pro
  - **Pro Plan**: Starting at $15-30/month
    - ~400 prompts per 5-hour cycle
    - ~5× the usage quota of Lite Plan
  - **Max Plan**: Higher tier
    - ~1600 prompts per 5-hour cycle
    - ~4× the usage quota of Pro Plan
- **Token Allowance**: Tens of billions of tokens monthly at ~1% of standard API pricing
- **Key Limitation**: Can ONLY be used within supported coding tools - NOT for direct API calls

#### B. Standard API (Pay-Per-Token)
- **Model**: Pure pay-as-you-go token-based pricing
- **Endpoint**: `https://api.z.ai/api/paas/v4/chat/completions`
- **Pricing Examples** (per 1M tokens):
  - GLM-4.7-Flash: $0.06 input / $0.40 output (cheapest)
  - GLM-4.5-Air: $0.13 input / $0.85 output
  - GLM-4.7: $0.40 input / $1.50 output
  - GLM-5: $0.75 input / $2.55 output
- **Billing**: Deducted from account balance or linked payment method

**Critical Distinction**: 
- Coding Plan quota does NOT apply to API calls
- API calls are billed separately and do not use Coding Plan quota
- If Coding Plan quota runs out, you wait for the next 5-hour refresh cycle
- API calls outside coding tools are NOT available to Coding Plan subscribers (only within supported tools)

---

### 2. Is there any "unlimited API" plan or bundle?

**Answer: NO - But Coding Plans Provide Massive Allowances**

- **No true unlimited API plan exists**
- However, the GLM Coding Plans provide **effectively unlimited usage** for coding scenarios:
  - Users report running 40-60M tokens per day on $3 plans without hitting limits
  - Monthly allowance: "tens of billions of tokens"
  - 5-hour cycle refresh system prevents hard caps
  - Weekly usage quota applies (but users who subscribed before Feb 12 have unlimited weekly usage)

**Important Note**: 
- These massive allowances are ONLY for coding tools
- Direct API usage is always pay-per-token
- No "unlimited API" bundle for general API access exists

---

### 3. What's the difference between their web chat subscription and API access billing?

**Answer: Z.ai Does NOT Have a Separate Web Chat Subscription**

**Finding**: Z.ai's chat interface (chat.z.ai) appears to be **free** without a separate paid web chat subscription tier (unlike ChatGPT Plus or Claude Plus).

**Current Structure**:
- **Web Chat (chat.z.ai)**: Free access to GLM models via web interface
- **API Access**: Pay-per-token billing for programmatic access
- **Coding Tools**: GLM Coding Plan subscription (hybrid model)

**No Web Chat Premium**: Unlike competitors (OpenAI's ChatGPT Plus at $20/month, Anthropic's Claude Pro), Z.ai does not appear to offer a separate paid subscription for web chat access. The web chat is free, and the paid tiers are:
1. GLM Coding Plan (for IDE/coding tools)
2. Standard API (for direct API integration)

---

## Pricing Comparison

### GLM Coding Plan Value Proposition
- Lite Plan: ~$6/month regular (3× Claude Pro quota at 1/3 the price)
- Provides 3-5× more usage than Claude Pro for significantly less cost
- Extremely cost-effective: ~1% of standard API pricing

### Standard API Pricing (Pay-Per-Token)
- Most affordable: GLM-4.7-Flash at $0.06/$0.40 per 1M tokens
- Mid-range: GLM-4.5-Air at $0.13/$0.85 per 1M tokens
- Premium: GLM-5 at $0.75/$2.55 per 1M tokens

---

## Important Clarifications

1. **Endpoint Matters**: Different endpoints for different billing:
   - Coding tools: `https://api.z.ai/api/anthropic` or `https://api.z.ai/api/coding/paas/v4`
   - Standard API: `https://api.z.ai/api/paas/v4/chat/completions`

2. **Quota Isolation**: Coding Plan quota is completely separate from account balance
   - Quota exhaustion doesn't trigger account balance deduction
   - API calls require separate account funding

3. **Model Support**: Not all models available in all plans
   - Lite Plan: GLM-4.7, GLM-4.6, GLM-4.5, GLM-4.5-Air
   - Pro/Max: Add GLM-5 support
   - Standard API: All models available

4. **Promotional Pricing**: Initial prices ($3/month) are promotional
   - Second cycle onwards: Regular pricing (typically 2× initial)
   - Annual plans available at better rates

---

## Sources
- Z.AI Official Documentation: https://docs.z.ai/guides/overview/pricing
- Z.AI Developer Documentation: https://docs.z.ai/devpack/overview
- Z.AI FAQ: https://docs.z.ai/devpack/faq
- Price Per Token: https://pricepertoken.com/pricing-page/provider/z-ai
- Cline Blog: https://cline.bot/blog/zai-cline-3-dollar-ai-coding
- Reddit discussions: r/vibecoding, r/CLine, r/ClaudeCode

---

## Confidence Assessment
**HIGH CONFIDENCE** - Information sourced from official Z.ai documentation, developer guides, and FAQ pages.

## Key Takeaway
Z.ai uses a **hybrid billing model**: 
- Subscription plans for coding tools (very affordable)
- Pay-per-token for standard API access
- No web chat premium subscription tier
- These two systems are completely separate and do not interact
