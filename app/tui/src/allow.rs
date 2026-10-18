//! The allow-rule chooser: the three scopes offered for a selected launch and the exact rule
//! text each would write. The daemon owns the rule syntax and the config file; this mirrors
//! `build_rule` in `src/jevmon/allow.py` (same escaping, same shapes) so the preview is the text
//! the daemon will store, and the daemon validates it again before writing.

/// What a rule covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// This exe with exactly these arguments.
    Exact,
    /// This exe, the first `k` arguments and anything after.
    Prefix,
    /// This exe with any arguments.
    Exe,
}

impl Scope {
    pub const ALL: [Scope; 3] = [Scope::Exact, Scope::Prefix, Scope::Exe];

    pub fn name(self) -> &'static str {
        match self {
            Scope::Exact => "exact",
            Scope::Prefix => "prefix",
            Scope::Exe => "exe only",
        }
    }
}

/// A literal fragment of a rule: `\` and `*` are escaped so they match themselves.
pub fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('*', "\\*")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chooser {
    pub event_id: u64,
    pub exe: String,
    /// argv[1:], split on whitespace (the daemon matches on the space-joined text).
    pub words: Vec<String>,
    pub scope: Scope,
    /// Arguments kept by the prefix scope, 1..=words.len().
    pub k: usize,
}

impl Chooser {
    /// A chooser for a launch (`exe` path, space-joined `argv` including argv[0]).
    pub fn new(event_id: u64, exe: &str, argv: &str) -> Result<Self, String> {
        if exe.is_empty() {
            return Err("this event has no exe to allow".to_string());
        }
        if !exe.starts_with('/') {
            return Err("the exe is not an absolute path".to_string());
        }
        if exe.chars().any(char::is_whitespace) {
            return Err("the exe path contains whitespace".to_string());
        }
        let words: Vec<String> = argv
            .split_whitespace()
            .skip(1)
            .map(str::to_string)
            .collect();
        let scope = if words.is_empty() {
            Scope::Exe
        } else {
            Scope::Prefix
        };
        Ok(Self {
            event_id,
            exe: exe.to_string(),
            words,
            scope,
            k: 1,
        })
    }

    /// The scopes on offer: the prefix scope needs at least one argument (with k = 0 it would be
    /// the exe-only rule again).
    pub fn scopes(&self) -> Vec<Scope> {
        Scope::ALL
            .into_iter()
            .filter(|s| *s != Scope::Prefix || !self.words.is_empty())
            .collect()
    }

    pub fn rule_for(&self, scope: Scope) -> String {
        let mut parts = vec![escape(&self.exe)];
        match scope {
            Scope::Exact => parts.extend(self.words.iter().map(|w| escape(w))),
            Scope::Prefix => {
                let k = self.k.clamp(1, self.words.len().max(1));
                parts.extend(self.words.iter().take(k).map(|w| escape(w)));
                parts.push("*".to_string());
            }
            Scope::Exe => parts.push("*".to_string()),
        }
        parts.join(" ")
    }

    /// The rule that Enter would add.
    pub fn rule(&self) -> String {
        self.rule_for(self.scope)
    }

    pub fn move_scope(&mut self, delta: isize) {
        let scopes = self.scopes();
        let at = scopes.iter().position(|s| *s == self.scope).unwrap_or(0) as isize;
        let next = (at + delta).rem_euclid(scopes.len() as isize) as usize;
        self.scope = scopes[next];
    }

    /// Change how many arguments the prefix scope keeps; selects that scope.
    pub fn adjust_k(&mut self, delta: isize) {
        if self.words.is_empty() {
            return;
        }
        self.scope = Scope::Prefix;
        let k = (self.k as isize + delta).clamp(1, self.words.len() as isize);
        self.k = k as usize;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIT: &str = "/nix/store/abc-git-2.55.0/bin/git";

    fn git() -> Chooser {
        Chooser::new(1, GIT, &format!("{GIT} -C /w/repo diff --name-only HEAD")).unwrap()
    }

    #[test]
    fn three_scopes_for_a_git_launch() {
        let c = git();
        assert_eq!(
            c.rule_for(Scope::Exact),
            format!("{GIT} -C /w/repo diff --name-only HEAD")
        );
        assert_eq!(c.rule_for(Scope::Prefix), format!("{GIT} -C *"));
        assert_eq!(c.rule_for(Scope::Exe), format!("{GIT} *"));
        assert_eq!(c.scope, Scope::Prefix);
    }

    #[test]
    fn k_is_clamped_and_never_zero() {
        let mut c = git();
        c.adjust_k(-5);
        assert_eq!(c.k, 1);
        c.adjust_k(99);
        assert_eq!(c.k, 5);
        assert_eq!(
            c.rule(),
            format!("{GIT} -C /w/repo diff --name-only HEAD *")
        );
    }

    #[test]
    fn no_arguments_offers_no_prefix_scope() {
        let mut c = Chooser::new(1, GIT, GIT).unwrap();
        assert_eq!(c.scopes(), [Scope::Exact, Scope::Exe]);
        assert_eq!(c.scope, Scope::Exe);
        assert_eq!(c.rule_for(Scope::Exact), GIT);
        c.adjust_k(1);
        assert_eq!(c.scope, Scope::Exe, "k does nothing without arguments");
    }

    #[test]
    fn literal_stars_and_backslashes_are_escaped() {
        let c = Chooser::new(1, "/bin/find", r"/bin/find . -name *.o a\b").unwrap();
        assert_eq!(c.rule_for(Scope::Exact), r"/bin/find . -name \*.o a\\b");
        assert_eq!(c.rule_for(Scope::Prefix), "/bin/find . *");
    }

    #[test]
    fn unbuildable_launches_are_refused() {
        assert!(Chooser::new(1, "", "x").is_err());
        assert!(Chooser::new(1, "git", "git status").is_err());
        assert!(Chooser::new(1, "/opt/my tool/x", "/opt/my tool/x").is_err());
    }

    #[test]
    fn scope_movement_wraps_over_the_offered_scopes() {
        let mut c = git();
        c.move_scope(1);
        assert_eq!(c.scope, Scope::Exe);
        c.move_scope(1);
        assert_eq!(c.scope, Scope::Exact);
        c.move_scope(-1);
        assert_eq!(c.scope, Scope::Exe);
    }
}
