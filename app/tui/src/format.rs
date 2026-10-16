//! Humane formatting of the daemon's raw numbers: rates, durations, percentages, timestamps.

/// `HH:MM:SS` in the machine's local time zone; `-` when the timestamp is unusable.
pub fn timestamp(epoch: f64) -> String {
    if !(0.0..=253_402_300_799.0).contains(&epoch) {
        return "-".to_string();
    }
    let secs = epoch as libc::time_t;
    // SAFETY: localtime_r only writes through the out parameter.
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        if libc::localtime_r(&secs, &mut tm).is_null() {
            return "-".to_string();
        }
        format!("{:02}:{:02}:{:02}", tm.tm_hour, tm.tm_min, tm.tm_sec)
    }
}

/// Seconds as `3m 20s`, `45s`, `1h 2m`, `0.5s`.
pub fn format_duration(secs: f64) -> String {
    let secs = if secs.is_finite() && secs > 0.0 {
        secs
    } else {
        0.0
    };
    if secs >= 3_600.0 {
        let h = (secs / 3_600.0).floor() as u64;
        let m = ((secs % 3_600.0) / 60.0).floor() as u64;
        format!("{h}h {m}m")
    } else if secs >= 60.0 {
        let m = (secs / 60.0).floor() as u64;
        let s = (secs % 60.0).floor() as u64;
        format!("{m}m {s}s")
    } else if secs < 10.0 && secs != secs.floor() {
        format!("{secs:.1}s")
    } else {
        format!("{:.0}s", secs.floor())
    }
}

/// A rate as `1.4/s`, `12/s`.
pub fn format_rate(value: f64) -> String {
    let value = if value.is_finite() && value >= 0.0 {
        value
    } else {
        0.0
    };
    if value < 10.0 {
        format!("{value:.1}/s")
    } else {
        format!("{value:.0}/s")
    }
}

/// A ratio as `9.8%` or `98%`.
pub fn format_percent(ratio: f64) -> String {
    let ratio = if ratio.is_finite() && ratio >= 0.0 {
        ratio
    } else {
        0.0
    };
    let pct = ratio * 100.0;
    if pct < 10.0 {
        format!("{pct:.1}%")
    } else {
        format!("{pct:.0}%")
    }
}

/// A poller interval as `50ms`, `1.4s`, or a long duration if it ever gets that big.
pub fn format_interval(secs: f64) -> String {
    let secs = if secs.is_finite() && secs > 0.0 {
        secs
    } else {
        0.0
    };
    if secs < 1.0 {
        format!("{:.0}ms", secs * 1_000.0)
    } else if secs < 60.0 {
        format!("{secs:.1}s")
    } else {
        format_duration(secs)
    }
}

/// `1,234`-style grouping for counters.
pub fn format_count(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_are_humane() {
        assert_eq!(format_duration(200.0), "3m 20s");
        assert_eq!(format_duration(45.0), "45s");
        assert_eq!(format_duration(3_725.0), "1h 2m");
        assert_eq!(format_duration(0.0), "0s");
        assert_eq!(format_duration(-5.0), "0s");
        assert_eq!(format_duration(0.5), "0.5s");
    }

    #[test]
    fn rates_and_percentages_round_kindly() {
        assert_eq!(format_rate(1.43), "1.4/s");
        assert_eq!(format_rate(12.4), "12/s");
        assert_eq!(format_rate(0.0), "0.0/s");
        assert_eq!(format_percent(0.098), "9.8%");
        assert_eq!(format_percent(0.98), "98%");
        assert_eq!(format_percent(0.0), "0.0%");
    }

    #[test]
    fn intervals_use_milliseconds_below_a_second() {
        assert_eq!(format_interval(0.05), "50ms");
        assert_eq!(format_interval(1.4), "1.4s");
        assert_eq!(format_interval(0.0), "0ms");
    }

    #[test]
    fn counts_group_thousands() {
        assert_eq!(format_count(999), "999");
        assert_eq!(format_count(1_234), "1,234");
        assert_eq!(format_count(12_345_678), "12,345,678");
    }

    #[test]
    fn timestamp_is_bounded_and_sane() {
        assert_eq!(timestamp(-1.0), "-");
        assert_eq!(timestamp(f64::NAN), "-");
        let text = timestamp(1_728_000_000.0); // 2024-10-05, machine-local
        assert_eq!(text.len(), 8);
        assert!(text.contains(':'));
    }
}
