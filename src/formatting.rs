pub fn format_duration_human(duration: &std::time::Duration) -> String {
    let sec_f = duration.as_secs_f32();
    if sec_f < 60. {
        let msec = (sec_f * 1000.).round() as u64;
        if sec_f > 0. && msec == 0 {
            "<1ms".to_string()
        } else if msec > 0 && duration.as_secs_f32() < 1. {
            format!("{}ms", msec)
        } else {
            format!("{:>3.1}s", sec_f)
        }
    } else {
        let sec = sec_f.round() as u64;
        let (min, sec) = (sec / 60, sec % 60);
        if min < 60 {
            format!("{}m{:02}s", min, sec)
        } else {
            let (hr, min) = (min / 60, min % 60);
            if hr < 24 {
                format!("{}h{:02}m{:02}s", hr, min, sec)
            } else {
                let (day, hr) = (hr / 24, hr % 24);
                format!("{}d{:02}h{:02}m{:02}s", day, hr, min, sec)
            }
        }
    }
}

pub fn format_duration(d: std::time::Duration) -> String {
    if d.as_secs_f32() < 60. {
        format!("{:>.1}sec", d.as_secs_f32())
    } else {
        format!(
            "{} ( {:>.1}sec )",
            format_duration_human(&d),
            d.as_secs_f32()
        )
    }
}
