# Eval Runner Summary

## Overview
The `runner.rs` file in the `eval` directory is the core runner for evaluation benchmarks in the nanobot project. It provides a suite of functions to evaluate LLMs across multiple tasks.

## Key Components

### 1. Hanoi Calibration
- **Purpose**: Measures LLM accuracy and latency on the Hanoi problem at varying disk counts
- **Process**: 
  - Samples steps of the Hanoi problem
  - Builds prompts with current disk states
  - Calls LLM for each step
  - Verifies correctness and computes metrics
- **Output**: `CalibrationResult` with accuracy, median latency, red-flag rate, and sample data

### 2. Hanoi Solve
- **Configuration**: `HanoiSolveConfig` struct
- **Features**: 
  - MAKER voting (multiple voters)
  - CATTS (confidence threshold) for answer acceptance
  - Disk count configuration

### 3. Sprint Runner
- **Purpose**: Run comprehensive evaluation sprints
- **Process**:
  - Generate knowledge corpus
  - Create questions from corpus
  - Run LLM on each question
  - Verify answers and compute scorecard

### 4. Learning Curve Runner
- **Purpose**: Evaluate learning progression over curriculum
- **Process**:
  - Execute tasks in difficulty sequence
  - Call LLM for each task
  - Verify answers and collect metrics

### 5. Haystack Aggregation Runner
- **Purpose**: Evaluate aggregation capabilities
- **Task Types**:
  - Counting
  - Distribution analysis
  - Filtering
  - Cross-referencing
  - Temporal analysis
- **Verification**: Case-insensitive content matching

## Testing
- Unit tests for:
  - Calibration result conversion
  - Default configurations
  - Mocked LLM calls

## Dependencies
- `serde`, `serde_json`, `chrono`, `uuid`, and internal crates