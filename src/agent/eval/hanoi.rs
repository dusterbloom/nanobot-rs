use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HanoiState {
    pub pegs: [Vec<u8>; 3],
    pub num_disks: u8,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HanoiMove {
    pub from: u8,
    pub to: u8,
}

impl HanoiState {
    /// Create initial state with all disks on peg 0
    /// Largest disk (n) at bottom, smallest (1) at top
    pub fn initial(n: u8) -> Self {
        let mut peg0 = Vec::new();
        for disk in (1..=n).rev() {
            peg0.push(disk);
        }
        HanoiState {
            pegs: [peg0, Vec::new(), Vec::new()],
            num_disks: n,
        }
    }

    /// Check if all disks are on peg 2 (goal state)
    pub fn is_goal(&self) -> bool {
        self.pegs[2].len() == self.num_disks as usize
            && self.pegs[0].is_empty()
            && self.pegs[1].is_empty()
    }

    /// Apply a move with validation
    pub fn apply_move(&mut self, m: &HanoiMove) -> Result<(), String> {
        if m.from > 2 || m.to > 2 {
            return Err(format!("Invalid peg index: from={}, to={}", m.from, m.to));
        }
        if m.from == m.to {
            return Err("Cannot move to same peg".to_string());
        }

        let from_idx = m.from as usize;
        let to_idx = m.to as usize;

        // Check if source peg is empty
        if self.pegs[from_idx].is_empty() {
            return Err(format!("Cannot move from empty peg {}", m.from));
        }

        let disk = self.pegs[from_idx].last().unwrap();

        // Check if we can place on destination (must be smaller than top disk)
        if let Some(&top_disk) = self.pegs[to_idx].last() {
            if disk > &top_disk {
                return Err(format!(
                    "Cannot place larger disk {} on smaller disk {}",
                    disk, top_disk
                ));
            }
        }

        // Perform the move
        let disk = self.pegs[from_idx].pop().unwrap();
        self.pegs[to_idx].push(disk);

        Ok(())
    }

    /// Get all valid moves from current state
    pub fn valid_moves(&self) -> Vec<HanoiMove> {
        let mut moves = Vec::new();

        for from in 0..3u8 {
            if self.pegs[from as usize].is_empty() {
                continue;
            }

            let disk = *self.pegs[from as usize].last().unwrap();

            for to in 0..3u8 {
                if from == to {
                    continue;
                }

                // Check if move is valid
                let can_move = if let Some(&top_disk) = self.pegs[to as usize].last() {
                    disk < top_disk
                } else {
                    true
                };

                if can_move {
                    moves.push(HanoiMove { from, to });
                }
            }
        }

        moves
    }

    /// Render ASCII display of state
    pub fn render(&self) -> String {
        let mut output = String::new();
        output.push_str("Towers of Hanoi State:\n");

        // Find max height
        let max_height = self.pegs.iter().map(|p| p.len()).max().unwrap_or(0);

        // Render from top to bottom
        for level in (0..max_height).rev() {
            for peg_idx in 0..3 {
                let peg = &self.pegs[peg_idx];
                if level < peg.len() {
                    output.push_str(&format!(" [{}] ", peg[level]));
                } else {
                    output.push_str("  |  ");
                }
            }
            output.push('\n');
        }

        // Base
        output.push_str("=====+=====+=====\n");
        output.push_str("  A      B      C\n");
        output.push_str("  0      1      2\n");

        output
    }
}

/// Generate optimal solution for n disks using recursive algorithm
pub fn optimal_solution(n: u8) -> Vec<HanoiMove> {
    let mut moves = Vec::new();
    hanoi_recursive(n, 0, 2, 1, &mut moves);
    moves
}

fn hanoi_recursive(n: u8, from: u8, to: u8, aux: u8, moves: &mut Vec<HanoiMove>) {
    if n == 0 {
        return;
    }
    if n == 1 {
        moves.push(HanoiMove { from, to });
        return;
    }

    // Move n-1 disks from 'from' to 'aux' using 'to'
    hanoi_recursive(n - 1, from, aux, to, moves);

    // Move disk n from 'from' to 'to'
    moves.push(HanoiMove { from, to });

    // Move n-1 disks from 'aux' to 'to' using 'from'
    hanoi_recursive(n - 1, aux, to, from, moves);
}

/// Replay optimal solution to step_idx, return (state_before, expected_move)
pub fn state_at_step(n: u8, step_idx: usize) -> (HanoiState, HanoiMove) {
    let solution = optimal_solution(n);
    if step_idx >= solution.len() {
        panic!(
            "step_idx {} out of bounds for solution length {}",
            step_idx,
            solution.len()
        );
    }

    let mut state = HanoiState::initial(n);
    for i in 0..step_idx {
        state.apply_move(&solution[i]).unwrap();
    }

    (state, solution[step_idx].clone())
}

/// Build LLM prompt from current state
pub fn build_prompt(state: &HanoiState) -> String {
    let mut prompt = String::new();
    prompt.push_str("Towers of Hanoi Problem\n\n");
    prompt.push_str("Rules:\n");
    prompt.push_str("1. Only one disk can be moved at a time\n");
    prompt.push_str("2. A disk can only be placed on top of a larger disk or an empty peg\n");
    prompt.push_str("3. Goal: move all disks from peg A (0) to peg C (2)\n\n");
    prompt.push_str("Current state:\n");
    prompt.push_str(&state.render());
    prompt.push_str("\n");
    prompt.push_str("What is the next move? Respond with the move in the format:\n");
    prompt.push_str("\"Move disk from peg X to peg Y\" (where X and Y are A/B/C or 0/1/2)\n");

    prompt
}

/// Parse move from LLM response
/// Handles formats: "A->C", "1->3", "peg 0 to peg 2", "Move disk from A to C", etc.
pub fn parse_move(response: &str) -> Option<HanoiMove> {
    let response = response.to_lowercase();

    // Try arrow format: "a->c" or "1->3"
    if let Some(arrow_pos) = response.find("->") {
        let before = response[..arrow_pos].trim();
        let after = response[arrow_pos + 2..].trim();

        if let (Some(from), Some(to)) = (parse_peg(before), parse_peg(after)) {
            return Some(HanoiMove { from, to });
        }
    }

    // Try "from X to Y" format
    if let Some(from_pos) = response.find("from") {
        if let Some(to_pos) = response.find("to") {
            if to_pos > from_pos {
                let from_part = &response[from_pos + 4..to_pos].trim();
                let to_part = &response[to_pos + 2..].trim();

                if let (Some(from), Some(to)) = (parse_peg(from_part), parse_peg(to_part)) {
                    return Some(HanoiMove { from, to });
                }
            }
        }
    }

    // Try finding peg references in order
    let pegs: Vec<u8> = response
        .split_whitespace()
        .filter_map(|word| parse_peg(word))
        .collect();

    if pegs.len() >= 2 {
        return Some(HanoiMove {
            from: pegs[0],
            to: pegs[1],
        });
    }

    None
}

fn parse_peg(s: &str) -> Option<u8> {
    let s = s.trim().to_lowercase();

    // Remove common prefixes
    let s = s
        .strip_prefix("peg")
        .unwrap_or(&s)
        .trim()
        .strip_prefix(":")
        .unwrap_or_else(|| {
            s.strip_prefix("peg")
                .unwrap_or(&s)
                .trim()
        });

    // Try letter format (A/B/C)
    if s.len() == 1 {
        let ch = s.chars().next()?;
        match ch {
            'a' => return Some(0),
            'b' => return Some(1),
            'c' => return Some(2),
            _ => {}
        }
    }

    // Try numeric format - treat as 1-indexed (1/2/3 -> 0/1/2)
    if let Ok(num) = s.parse::<u8>() {
        if num >= 1 && num <= 3 {
            return Some(num - 1); // Convert 1-indexed to 0-indexed
        }
    }

    None
}

/// Deterministic strided sampling of n indices from 0..total
pub fn sample_indices(total: usize, n: usize) -> Vec<usize> {
    if n == 0 || total == 0 {
        return Vec::new();
    }
    if n >= total {
        return (0..total).collect();
    }

    let stride = total as f64 / n as f64;
    (0..n).map(|i| (i as f64 * stride) as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let state = HanoiState::initial(3);
        assert_eq!(state.pegs[0], vec![3u8, 2, 1]);
        assert_eq!(state.pegs[1], Vec::<u8>::new());
        assert_eq!(state.pegs[2], Vec::<u8>::new());
        assert_eq!(state.num_disks, 3);
    }

    #[test]
    fn test_initial_single_disk() {
        let state = HanoiState::initial(1);
        assert_eq!(state.pegs[0], vec![1u8]);
        assert_eq!(state.pegs[1], Vec::<u8>::new());
        assert_eq!(state.pegs[2], Vec::<u8>::new());
    }

    #[test]
    fn test_is_goal() {
        let mut state = HanoiState::initial(3);
        assert!(!state.is_goal());

        state.pegs[0].clear();
        state.pegs[2] = vec![3, 2, 1];
        assert!(state.is_goal());
    }

    #[test]
    fn test_is_goal_not_reached() {
        let mut state = HanoiState::initial(3);
        state.pegs[0] = vec![3, 2];
        state.pegs[2] = vec![1];
        assert!(!state.is_goal());
    }

    #[test]
    fn test_apply_valid_move() {
        let mut state = HanoiState::initial(3);
        let result = state.apply_move(&HanoiMove { from: 0, to: 1 });
        assert!(result.is_ok());
        assert_eq!(state.pegs[0], vec![3, 2]);
        assert_eq!(state.pegs[1], vec![1]);
    }

    #[test]
    fn test_apply_invalid_move_empty_peg() {
        let mut state = HanoiState::initial(3);
        let result = state.apply_move(&HanoiMove { from: 1, to: 2 });
        assert!(result.is_err());
    }

    #[test]
    fn test_apply_invalid_move_larger_on_smaller() {
        let mut state = HanoiState::initial(3);
        state.apply_move(&HanoiMove { from: 0, to: 1 }).unwrap(); // Move disk 1 to peg 1
        let result = state.apply_move(&HanoiMove { from: 0, to: 1 }); // Try to move disk 2 onto disk 1
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_moves_initial() {
        let state = HanoiState::initial(3);
        let moves = state.valid_moves();
        assert_eq!(moves.len(), 2); // Can move smallest disk to peg 1 or 2
    }

    #[test]
    fn test_optimal_solution_1_disk() {
        let solution = optimal_solution(1);
        assert_eq!(solution.len(), 1);
        assert_eq!(solution[0], HanoiMove { from: 0, to: 2 });
    }

    #[test]
    fn test_optimal_solution_3_disks() {
        let solution = optimal_solution(3);
        assert_eq!(solution.len(), 7); // 2^3 - 1 = 7

        // Verify by replaying
        let mut state = HanoiState::initial(3);
        for move_step in &solution {
            assert!(state.apply_move(move_step).is_ok());
        }
        assert!(state.is_goal());
    }

    #[test]
    fn test_optimal_solution_reaches_goal() {
        let n = 5;
        let solution = optimal_solution(n);
        assert_eq!(solution.len(), 31); // 2^5 - 1 = 31

        let mut state = HanoiState::initial(n);
        for move_step in &solution {
            state.apply_move(move_step).unwrap();
        }
        assert!(state.is_goal());
    }

    #[test]
    fn test_state_at_step() {
        let (state, next_move) = state_at_step(3, 0);
        assert_eq!(state.pegs[0], vec![3, 2, 1]);
        assert_eq!(next_move, HanoiMove { from: 0, to: 2 });

        let (state, next_move) = state_at_step(3, 1);
        assert_eq!(state.pegs[0], vec![3, 2]);
        assert_eq!(state.pegs[2], vec![1]);
        assert_eq!(next_move, HanoiMove { from: 0, to: 1 });
    }

    #[test]
    fn test_parse_move_arrow_format() {
        let result = parse_move("A->C");
        assert_eq!(result, Some(HanoiMove { from: 0, to: 2 }));

        let result = parse_move("a->b");
        assert_eq!(result, Some(HanoiMove { from: 0, to: 1 }));
    }

    #[test]
    fn test_parse_move_numeric() {
        let result = parse_move("1->3");
        assert_eq!(result, Some(HanoiMove { from: 0, to: 2 }));

        let result = parse_move("2->1");
        assert_eq!(result, Some(HanoiMove { from: 1, to: 0 }));
    }

    #[test]
    fn test_parse_move_verbose() {
        let result = parse_move("Move disk from peg A to peg C");
        assert_eq!(result, Some(HanoiMove { from: 0, to: 2 }));

        let result = parse_move("from peg 1 to peg 3");
        assert_eq!(result, Some(HanoiMove { from: 0, to: 2 }));

        let result = parse_move("Move from B to A");
        assert_eq!(result, Some(HanoiMove { from: 1, to: 0 }));
    }

    #[test]
    fn test_parse_move_invalid() {
        let result = parse_move("this is garbage");
        assert_eq!(result, None);

        let result = parse_move("move disk");
        assert_eq!(result, None);
    }

    #[test]
    fn test_sample_indices() {
        let indices = sample_indices(10, 3);
        assert_eq!(indices.len(), 3);
        assert!(indices.iter().all(|&i| i < 10));

        let indices = sample_indices(100, 10);
        assert_eq!(indices.len(), 10);
        assert_eq!(indices[0], 0);
        assert!(indices[9] < 100);
    }

    #[test]
    fn test_render_basic() {
        let state = HanoiState::initial(3);
        let rendered = state.render();
        assert!(!rendered.is_empty());
        assert!(rendered.contains("Towers of Hanoi State"));
        assert!(rendered.contains("A"));
        assert!(rendered.contains("B"));
        assert!(rendered.contains("C"));
    }

    #[test]
    fn test_build_prompt_contains_state() {
        let state = HanoiState::initial(3);
        let prompt = build_prompt(&state);
        assert!(prompt.contains("Towers of Hanoi"));
        assert!(prompt.contains("Rules"));
        assert!(prompt.contains("peg"));
    }
}
