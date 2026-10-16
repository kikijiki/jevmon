//! The event list's view state: which rows are shown, which event is selected and where the
//! viewport sits — a small, pure state machine so the follow-tail rules can be tested without a
//! terminal.
//!
//! The rules (non-negotiable):
//!
//! - IMPLICIT TAIL: until the user navigates there is NO pinned selection. Follow is on, the
//!   view always shows the newest rows (autoscroll) and `selected_id()` is the tail — so the
//!   detail pane and the highlight track the newest event.
//! - PINNING: any explicit navigation (j/k, arrows, pgup/pgdn, home, wheel up, clicking a row)
//!   pins the selection to an event id and stops following. The view parks; new events are
//!   appended without moving the view or the selection.
//! - RETURNING TO THE BOTTOM (End/G, navigating onto the last row, clicking the newest row,
//!   wheel-scrolling to the bottom) clears the pin and resumes following — so a pinned
//!   selection is never hidden by the tail. The pin is tracked by event id, never by row index,
//!   so it survives appended rows, filter rebuilds (falling to the neighbour row if filtered
//!   out) and cache pruning.

/// View state for the scrollable list of event ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListState {
    rows: Vec<u64>,
    /// The user's pinned selection; `None` means the implicit tail (while following).
    selected: Option<u64>,
    offset: usize,
    viewport: usize,
    follow: bool,
}

impl Default for ListState {
    fn default() -> Self {
        Self::new()
    }
}

impl ListState {
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            selected: None,
            offset: 0,
            viewport: 10,
            follow: true,
        }
    }

    pub fn rows(&self) -> &[u64] {
        &self.rows
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The event the view is "on": the pinned selection, or the tail while following.
    pub fn selected_id(&self) -> Option<u64> {
        if let Some(id) = self.selected {
            Some(id)
        } else if self.follow {
            self.rows.last().copied()
        } else {
            None
        }
    }

    /// The user's pinned selection, if any (`None` while following the implicit tail).
    pub fn pinned_id(&self) -> Option<u64> {
        self.selected
    }

    pub fn selected_index(&self) -> Option<usize> {
        self.selected_id()
            .and_then(|id| self.rows.iter().position(|&r| r == id))
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn viewport(&self) -> usize {
        self.viewport
    }

    pub fn following(&self) -> bool {
        self.follow
    }

    pub fn at_bottom(&self) -> bool {
        self.rows.is_empty() || self.offset + self.viewport >= self.rows.len()
    }

    pub fn set_viewport(&mut self, height: usize) {
        let height = height.max(1);
        if height != self.viewport {
            self.viewport = height;
            self.clamp();
        }
    }

    /// Replace the visible rows (filter change, prune). A pinned selection follows the event
    /// id; if it is filtered out, the selection falls to whatever took its place so the user is
    /// not dumped at the top of the list.
    pub fn set_rows(&mut self, rows: Vec<u64>) {
        let old_index = self
            .pinned_id()
            .and_then(|id| self.rows.iter().position(|&r| r == id));
        self.rows = rows;
        if let Some(id) = self.selected
            && !self.rows.contains(&id)
        {
            self.selected = old_index
                .and_then(|i| self.rows.get(i.min(self.rows.len().saturating_sub(1))))
                .copied();
        }
        self.clamp();
    }

    /// Select an event id: the tail row resumes following, anything else pins.
    pub fn select_id(&mut self, id: u64) {
        if !self.rows.contains(&id) {
            return;
        }
        if self.rows.last() == Some(&id) {
            self.follow_tail();
        } else {
            self.pin(id);
        }
    }

    pub fn select_prev(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let index = self.selected_index().unwrap_or(self.rows.len());
        self.select_index(index.saturating_sub(1));
    }

    pub fn select_next(&mut self) {
        if self.rows.is_empty() {
            return;
        }
        let index = self.selected_index().unwrap_or(0);
        if index + 1 < self.rows.len() {
            self.select_index(index + 1);
        } else if self.selected.is_none() {
            // Explicit Down at the implicit tail: pin the newest row and stop the follow.
            if let Some(id) = self.rows.get(index).copied() {
                self.pin(id);
            }
        }
        // Pinned at the last row already: nowhere to move.
    }

    pub fn select_home(&mut self) {
        self.select_index(0);
    }

    pub fn select_end(&mut self) {
        self.follow_tail();
    }

    /// Page up/down: move the selection by one screenful.
    pub fn select_page(&mut self, forward: bool) {
        if self.rows.is_empty() {
            return;
        }
        let page = self.viewport.saturating_sub(1).max(1);
        let index = self.selected_index().unwrap_or(0);
        let target = if forward {
            (index + page).min(self.rows.len() - 1)
        } else {
            index.saturating_sub(page)
        };
        self.select_index(target);
    }

    /// Scroll the viewport (wheel). Scrolling away pins whatever is selected (the tail, when
    /// implicit) and stops following; scrolling back to the bottom resumes it and clears the pin.
    pub fn scroll_by(&mut self, delta: isize) {
        if self.rows.is_empty() {
            self.follow = true;
            return;
        }
        let max = self.rows.len().saturating_sub(self.viewport);
        let next = self.offset as isize + delta;
        self.offset = next.clamp(0, max as isize) as usize;
        if self.at_bottom() {
            self.follow_tail();
        } else {
            if self.selected.is_none() {
                self.selected = self.rows.last().copied();
            }
            self.follow = false;
        }
    }

    fn select_index(&mut self, index: usize) {
        if self.rows.is_empty() {
            return;
        }
        if index >= self.rows.len() - 1 {
            self.follow_tail();
        } else if let Some(id) = self.rows.get(index).copied() {
            self.pin(id);
        }
    }

    fn pin(&mut self, id: u64) {
        self.selected = Some(id);
        self.ensure_selected_visible();
        self.follow = false;
    }

    /// Back to the implicit tail: clear the pin, follow the newest rows.
    fn follow_tail(&mut self) {
        self.selected = None;
        self.follow = true;
        self.offset = self.rows.len().saturating_sub(self.viewport);
    }

    fn ensure_selected_visible(&mut self) {
        if let Some(i) = self.selected_index() {
            if i < self.offset {
                self.offset = i;
            } else if i >= self.offset + self.viewport {
                self.offset = i + 1 - self.viewport;
            }
        }
    }

    fn clamp(&mut self) {
        let max = self.rows.len().saturating_sub(self.viewport);
        if self.follow && !self.rows.is_empty() {
            self.offset = max;
        }
        self.offset = self.offset.min(max);
    }
}

/// Convenience for the UI: the id range currently in view.
pub fn visible_rows(state: &ListState) -> &[u64] {
    let start = state.offset().min(state.len());
    let end = (start + state.viewport()).min(state.len());
    &state.rows()[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(range: std::ops::RangeInclusive<u64>) -> Vec<u64> {
        range.collect()
    }

    fn state_with(rows: Vec<u64>, viewport: usize) -> ListState {
        let mut state = ListState::new();
        state.set_viewport(viewport);
        state.set_rows(rows);
        state
    }

    #[test]
    fn first_load_selects_the_tail_and_follows() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_end();
        assert_eq!(state.selected_id(), Some(25));
        assert_eq!(state.pinned_id(), None);
        assert_eq!(state.offset(), 15);
        assert!(state.following());
        assert!(state.at_bottom());
    }

    #[test]
    fn following_scrolls_the_view_and_the_implicit_tail_tracks_it() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_end();
        // New events arrive at the tail: the view follows and the implicit selection is the
        // newest row — there is nothing pinned for autoscroll to fight.
        state.set_rows(ids(1..=30));
        assert_eq!(state.selected_id(), Some(30));
        assert_eq!(state.pinned_id(), None);
        assert_eq!(state.offset(), 20);
        assert!(state.at_bottom());
        assert!(state.following());
    }

    #[test]
    fn navigation_pins_the_selection_and_freezes_the_view() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_prev(); // Up from the implicit tail pins the second-newest row
        assert_eq!(state.selected_id(), Some(24));
        assert_eq!(state.pinned_id(), Some(24));
        assert!(!state.following());
        let offset = state.offset();
        state.set_rows(ids(1..=200)); // a long stream arrives
        assert_eq!(state.selected_id(), Some(24)); // pinned: the stream moves nothing
        assert_eq!(state.offset(), offset);
        state.select_next();
        assert_eq!(state.selected_id(), Some(25)); // Down steps exactly one row
        state.select_prev();
        assert_eq!(state.selected_id(), Some(24));
    }

    #[test]
    fn down_at_the_implicit_tail_pins_the_newest_row() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_end();
        state.select_next(); // cannot move down: pins the tail instead
        assert_eq!(state.selected_id(), Some(25));
        assert!(!state.following());
        state.set_rows(ids(1..=26));
        assert_eq!(state.selected_id(), Some(25));
        assert_eq!(state.offset(), 15);
    }

    #[test]
    fn scrolling_up_stops_following_and_freezes_the_view() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_end();
        state.scroll_by(-20);
        assert!(!state.following());
        assert_eq!(state.selected_id(), Some(25)); // wheel-up pins the tail it scrolled from
        assert_eq!(state.offset(), 0);
        let offset = state.offset();
        state.set_rows(ids(1..=40));
        assert_eq!(state.offset(), offset); // the view stayed where the user put it
        assert_eq!(state.selected_id(), Some(25));
        assert_eq!(state.selected_index(), Some(24));
    }

    #[test]
    fn scrolling_back_to_the_bottom_resumes_following() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_end();
        state.scroll_by(-100);
        assert!(!state.following());
        state.scroll_by(100);
        assert!(state.following());
        assert_eq!(state.pinned_id(), None); // back to the implicit tail
        state.set_rows(ids(1..=50));
        assert_eq!(state.offset(), 40); // autoscroll: the view rides the tail again
        assert_eq!(state.selected_id(), Some(50));
    }

    #[test]
    fn navigating_onto_the_last_row_resumes_following() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_home(); // pins the first row
        assert!(!state.following());
        assert_eq!(state.pinned_id(), Some(1));
        state.select_index(state.len() - 1); // e.g. pgdn landing on the last row
        assert!(state.following());
        assert_eq!(state.pinned_id(), None);
        assert_eq!(state.selected_id(), Some(25));
        state.select_end(); // End / G
        assert!(state.following());
        state.select_id(25); // clicking/navigating onto the tail row
        assert!(state.following());
        assert_eq!(state.pinned_id(), None);
    }

    #[test]
    fn selection_survives_a_filter_rebuild_by_id() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_id(22);
        // Filter keeps only even ids: 22 must stay selected at its new index.
        let even: Vec<u64> = (1..=25).filter(|id| id % 2 == 0).collect();
        state.set_rows(even);
        assert_eq!(state.selected_id(), Some(22));
        assert_eq!(state.selected_index(), Some(10));
    }

    #[test]
    fn selection_falls_to_the_neighbour_when_filtered_out() {
        let mut state = state_with(ids(1..=25), 10);
        state.select_id(22);
        let odd: Vec<u64> = (1..=25).filter(|id| id % 2 == 1).collect();
        state.set_rows(odd);
        assert_eq!(state.selected_id(), Some(25)); // the row that kept index 10's place
        let mut empty = state_with(ids(1..=5), 10);
        empty.select_id(3);
        empty.set_rows(Vec::new());
        assert_eq!(empty.selected_id(), None);
    }

    #[test]
    fn navigation_scrolls_minimally_and_tracks_following() {
        let mut state = state_with(ids(1..=50), 10);
        state.select_home();
        assert_eq!(state.selected_id(), Some(1));
        assert_eq!(state.offset(), 0);
        assert!(!state.following());
        state.select_next();
        assert_eq!(state.selected_id(), Some(2));
        state.select_page(true);
        assert_eq!(state.selected_id(), Some(11));
        assert_eq!(state.offset(), 1); // scrolled just enough to keep 11 visible
        state.select_end();
        assert!(state.offset() + 10 >= 50);
        assert!(state.following());
        state.select_prev();
        assert_eq!(state.selected_id(), Some(49));
        assert!(!state.following()); // explicit navigation pins and stops the follow
    }

    #[test]
    fn page_and_home_work_on_an_empty_list() {
        let mut state = ListState::new();
        state.select_page(true);
        state.select_home();
        state.select_end();
        state.select_next();
        assert_eq!(state.selected_id(), None);
    }

    #[test]
    fn viewport_resize_keeps_the_tail_in_view_while_following() {
        let mut state = state_with(ids(1..=30), 10);
        state.select_end();
        state.set_viewport(20);
        assert_eq!(state.offset(), 10);
        state.set_viewport(5);
        assert_eq!(state.offset(), 25);
    }

    #[test]
    fn visible_rows_slice_matches_the_offset() {
        let mut state = state_with(ids(1..=25), 10);
        state.scroll_by(-10); // following from offset 15 lands on 5
        assert_eq!(visible_rows(&state), &[6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
    }
}
