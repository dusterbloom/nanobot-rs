# Adaptive RLM Control System Design

## Executive Summary

**Recommendation: Adaptive Time-Boxed Execution with Dynamic Round Budgeting**

Use a **deadline-based scheduling approach** where the Director allocates a time budget to the RLM, and the RLM self-regulates iteration depth using exponential moving average (EMA) latency prediction.

## Core Mechanism

### 1. Data Structures

```python
@dataclass
class RLMBudget:
    """Budget allocated by Director to RLM for a delegated task"""
    time_budget_ms: int          # Hard deadline (e.g., 5000ms)
    min_rounds: int = 2          # Minimum rounds before stopping (quality floor)
    confidence_threshold: float = 0.85  # Stop if task confidence >= this
    
@dataclass
class RLMMetrics:
    """Runtime metrics tracked by RLM executor"""
    round_latencies: deque[int]  # Last N round latencies (ms), maxlen=10
    ema_latency: float           # Exponential moving average of latency
    alpha: float = 0.3           # EMA smoothing factor (weights recent rounds)
    rounds_completed: int = 0
    time_elapsed_ms: int = 0
    
    def update(self, round_latency_ms: int):
        self.round_latencies.append(round_latency_ms)
        if self.ema_latency == 0:
            self.ema_latency = round_latency_ms
        else:
            self.ema_latency = self.alpha * round_latency_ms + (1 - self.alpha) * self.ema_latency
        self.rounds_completed += 1
        self.time_elapsed_ms += round_latency_ms
    
    def predict_next_round_ms(self) -> int:
        """Predict next round latency with 20% safety margin"""
        return int(self.ema_latency * 1.2)
    
    def can_fit_another_round(self, budget: RLMBudget) -> bool:
        """Check if another round fits in remaining budget"""
        remaining = budget.time_budget_ms - self.time_elapsed_ms
        predicted = self.predict_next_round_ms()
        return predicted <= remaining

@dataclass
class TaskConfidence:
    """RLM's self-assessed confidence in task completion"""
    score: float  # 0.0 to 1.0
    reasoning: str  # Why this confidence level
```

### 2. Director Budget Allocation Algorithm

```python
class Director:
    def allocate_rlm_budget(self, task_complexity: str, user_latency_tolerance: str) -> RLMBudget:
        """
        Director decides budget based on:
        - Task complexity (simple/medium/complex)
        - User latency tolerance (interactive/background)
        
        This is the ONLY manual configuration point - simple heuristics
        """
        base_budgets = {
            "simple": 3000,    # 3 seconds
            "medium": 8000,    # 8 seconds  
            "complex": 20000,  # 20 seconds
        }
        
        budget_ms = base_budgets.get(task_complexity, 8000)
        
        # Interactive tasks get 50% budget reduction
        if user_latency_tolerance == "interactive":
            budget_ms = int(budget_ms * 0.5)
        
        # Minimum rounds scales with complexity
        min_rounds = {"simple": 1, "medium": 2, "complex": 3}.get(task_complexity, 2)
        
        return RLMBudget(
            time_budget_ms=budget_ms,
            min_rounds=min_rounds,
            confidence_threshold=0.85
        )
```

### 3. RLM Self-Regulation Loop

```python
class RLMExecutor:
    def execute_delegated_task(self, task: Task, budget: RLMBudget) -> Result:
        """
        RLM executes rounds until:
        1. Time budget exhausted (hard stop)
        2. Confidence threshold met AND min_rounds completed (early success)
        3. No progress detected for 3 consecutive rounds (stall detection)
        """
        metrics = RLMMetrics()
        scratch_pad = {}
        stall_counter = 0
        
        while True:
            round_start = time.monotonic_ns()
            
            # Execute one round (tool calls + analysis)
            result = self._execute_round(task, scratch_pad, metrics, budget)
            
            round_latency_ms = (time.monotonic_ns() - round_start) // 1_000_000
            metrics.update(round_latency_ms)
            
            # Stopping condition 1: Minimum rounds not met - keep going
            if metrics.rounds_completed < budget.min_rounds:
                continue
            
            # Stopping condition 2: Confidence threshold met - early exit
            if result.confidence.score >= budget.confidence_threshold:
                return result
            
            # Stopping condition 3: Stall detection (no progress)
            if self._is_stalled(result, scratch_pad):
                stall_counter += 1
                if stall_counter >= 3:
                    return result  # Give up, return best effort
            else:
                stall_counter = 0
            
            # Stopping condition 4: Budget exhausted
            if not metrics.can_fit_another_round(budget):
                return result
            
            # Continue to next round
    
    def _execute_round(self, task: Task, scratch_pad: dict, 
                       metrics: RLMMetrics, budget: RLMBudget) -> RoundResult:
        """
        Single round execution with adaptive token allocation.
        
        Key insight: Allocate more tokens when we have time budget remaining,
        fewer tokens when running low on time.
        """
        # Adaptive token budget based on remaining time
        time_remaining_ratio = (budget.time_budget_ms - metrics.time_elapsed_ms) / budget.time_budget_ms
        
        if time_remaining_ratio > 0.7:
            max_tokens = 2048  # Generous when plenty of time
        elif time_remaining_ratio > 0.3:
            max_tokens = 1024  # Standard
        else:
            max_tokens = 512   # Terse when low on time
        
        # Build prompt with scratch pad context
        prompt = self._build_prompt(task, scratch_pad, metrics, budget)
        
        # Call RLM (single-turn, no history)
        response = self.rlm.generate(prompt, max_tokens=max_tokens)
        
        # Parse response: tool calls, mem_store operations, confidence assessment
        parsed = self._parse_response(response)
        
        # Execute tool calls
        tool_results = self._execute_tools(parsed.tool_calls)
        
        # Update scratch pad
        scratch_pad.update(parsed.mem_stores)
        
        # RLM self-assesses confidence
        confidence = self._assess_confidence(parsed, tool_results, scratch_pad)
        
        return RoundResult(
            tool_results=tool_results,
            confidence=confidence,
            scratch_pad_state=scratch_pad.copy()
        )
    
    def _assess_confidence(self, parsed, tool_results, scratch_pad) -> TaskConfidence:
        """
        RLM self-assesses confidence in task completion.
        
        This is embedded in the RLM prompt template:
        "After executing tools, assess your confidence that the task is complete.
         Score 0.0-1.0 and explain your reasoning."
        
        The RLM learns to be calibrated through few-shot examples.
        """
        # Extract from parsed response (RLM outputs structured confidence)
        return TaskConfidence(
            score=parsed.confidence_score,
            reasoning=parsed.confidence_reasoning
        )
    
    def _is_stalled(self, result: RoundResult, scratch_pad: dict) -> bool:
        """
        Detect if RLM is making progress.
        
        Heuristics:
        - No new scratch pad entries in last round
        - Confidence score hasn't increased by 0.05+ in last 2 rounds
        - Same tool called 3+ times with no new information
        """
        # Implementation depends on tracking previous rounds
        # Simplified here for clarity
        return False  # Placeholder
```

### 4. Prompt Template for RLM Confidence Assessment

```python
RLM_ROUND_PROMPT = """
You are executing round {round_num} of a delegated task.

TASK: {task_description}

TIME BUDGET: {time_remaining_ms}ms remaining of {total_budget_ms}ms
ROUNDS COMPLETED: {rounds_completed}
MINIMUM ROUNDS: {min_rounds}

SCRATCH PAD (your persistent memory):
{scratch_pad_json}

INSTRUCTIONS:
1. Analyze the task and scratch pad
2. Call tools as needed (you have {max_tokens} tokens this round)
3. Store findings in scratch pad using mem_store
4. Assess your confidence in task completion (0.0-1.0)

OUTPUT FORMAT:
<tool_calls>
[your tool calls here]
</tool_calls>

<mem_store>
[key-value pairs to remember]
</mem_store>

<confidence>
score: [0.0-1.0]
reasoning: [why this score - be honest about uncertainty]
</confidence>

CALIBRATION EXAMPLES:
- 0.3: "Retrieved data but haven't analyzed patterns yet"
- 0.7: "Found answer but need to verify edge cases"  
- 0.9: "Verified answer, confident but allowing for rare errors"
- 1.0: "Mathematically certain (rare - use only for deterministic tasks)"
"""
```

## Why This Design Works

### 1. **Automatic Adaptation to Latency**
- **Cloud RLM (1s/round)**: EMA quickly stabilizes at ~1s, allows 5-8 rounds in 8s budget
- **Local RLM (3-5s/round)**: EMA stabilizes at ~4s, allows 2-3 rounds in 8s budget
- **No manual configuration needed** - system adapts in first 2-3 rounds

### 2. **Cost-Aware Without Explicit Pricing**
- Time budget implicitly captures cost (faster models = more rounds = higher cost acceptable)
- Director sets budget based on task value, not deployment mode
- Works for mixed setups (local director + cloud RLM) automatically

### 3. **Quality Floor via min_rounds**
- Prevents premature termination on first lucky guess
- Scales with task complexity, not deployment mode

### 4. **RLM Self-Regulation via Confidence**
- RLM learns to self-assess through few-shot calibration
- Early exit when confident (saves budget for other tasks)
- Honest uncertainty keeps it working when needed

### 5. **Graceful Degradation**
- Stall detection prevents infinite loops
- Adaptive token allocation preserves budget when running low
- Always returns best-effort result, never fails

### 6. **Forward-Compatible**
- New faster model? EMA adapts, allows more rounds automatically
- New slower model? EMA adapts, does fewer rounds
- Mixed deployments? Each RLM tracks its own metrics
- Multi-tenancy? Each task gets independent budget

## Implementation Checklist

1. **Minimal Configuration** (one-time setup):
   ```python
   # This is the ONLY config file needed
   TASK_COMPLEXITY_HEURISTICS = {
       "data_analysis": "medium",
       "simple_query": "simple", 
       "research": "complex",
   }
   
   USER_LATENCY_TOLERANCE = "interactive"  # or "background"
   ```

2. **RLM Prompt Engineering**:
   - Add confidence assessment to prompt template
   - Provide 10-15 calibration examples
   - Fine-tune confidence threshold (0.85) based on empirical testing

3. **Metrics Collection**:
   - Log (task_type, budget_allocated, rounds_completed, time_elapsed, confidence_final)
   - Use for offline analysis and budget heuristic tuning

4. **Optional Enhancements**:
   - **Adaptive alpha**: Increase EMA smoothing factor if latency variance is high
   - **Budget rollover**: Unused budget from fast tasks → pool for slow tasks
   - **Director feedback loop**: Track RLM success rate per task type, adjust budgets

## Comparison to Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Config Profiles** | Simple | Doesn't scale, manual tuning |
| **Fixed Iteration Count** | Predictable | Wastes budget or underperforms |
| **Token Budget Only** | Cost-aware | Ignores latency (bad UX) |
| **Pure Self-Regulation** | Elegant | RLM may not be calibrated |
| **This Design** | Adapts automatically, quality floor, forward-compatible | Requires confidence calibration |

## Theoretical Foundation

This design draws from:

1. **Real-Time Systems**: Deadline scheduling (EDF - Earliest Deadline First)
2. **Control Theory**: Exponential moving average for noise reduction in latency prediction
3. **Distributed Systems**: Time budgets over iteration counts (Google's "deadline propagation")
4. **Machine Learning**: Confidence calibration (Platt scaling, temperature scaling)

The key insight: **Time is the universal currency** that captures both cost and user experience. By making time the primary control variable and letting the RLM self-regulate within that budget, we get automatic adaptation without manual tuning.

---

## Example Scenarios

### Scenario 1: Cloud Deployment
- Director allocates 8000ms for medium task
- RLM round 1: 1100ms (EMA = 1100ms)
- RLM round 2: 950ms (EMA = 1055ms)
- RLM round 3: 1000ms (EMA = 1039ms, confidence = 0.60)
- RLM round 4: 980ms (EMA = 1021ms, confidence = 0.88)
- **Stops at round 4** (confidence >= 0.85, min_rounds met)
- Total time: 4030ms, 6 rounds remaining capacity unused (saved for other tasks)

### Scenario 2: Local Deployment (Same Task)
- Director allocates 8000ms for medium task (same budget!)
- RLM round 1: 4200ms (EMA = 4200ms)
- RLM round 2: 3800ms (EMA = 4080ms, confidence = 0.70)
- Predicted next round: 4896ms, remaining: 0ms
- **Stops at round 2** (budget exhausted)
- Total time: 8000ms, returns best-effort result

### Scenario 3: Local with Better Hardware
- Director allocates 8000ms for medium task
- RLM round 1: 2100ms (EMA = 2100ms, new GPU!)
- RLM round 2: 1950ms (EMA = 2055ms, confidence = 0.65)
- RLM round 3: 2000ms (EMA = 2039ms, confidence = 0.89)
- **Stops at round 3** (confidence >= 0.85)
- Total time: 6050ms
- **No code changes, automatically used faster hardware**

