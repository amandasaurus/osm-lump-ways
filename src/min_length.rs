#[derive(Clone, Debug, PartialEq)]
pub enum MinLengthFilter {
    Length(f64),
    PercentLongest(f64),
    IncludeTotalPercentage(f64),
    PercentTotal(f64),
}

impl std::str::FromStr for MinLengthFilter {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.to_ascii_lowercase();
        if let Ok(len) = s.parse::<f64>() {
            Ok(MinLengthFilter::Length(len))
        } else if let Some(kmstr) = s.strip_suffix("km")
            && let Ok(km) = kmstr.parse::<f64>()
        {
            Ok(MinLengthFilter::Length(km * 1000.))
        } else if let Some(kmstr) = s.strip_suffix("k")
            && let Ok(_) = kmstr.parse::<f64>()
        {
            Err(format!(
                "The raw suffix `k` is not accepted. Use `km`. i.e. ` --min-length {}km `",
                kmstr
            ))
        } else if let Some(mstr) = s.strip_suffix("m")
            && let Ok(m) = mstr.parse::<f64>()
        {
            Ok(MinLengthFilter::Length(m))
        } else if let Some(perc) = s.strip_suffix("%longest")
            && let Ok(perc) = perc.parse::<f64>()
        {
            Ok(MinLengthFilter::PercentLongest(perc / 100.))
        } else if let Some(perc0) = s.strip_prefix("incl")
            && let Some(perc) = perc0.strip_suffix("%total")
            && let Ok(perc) = perc.parse::<f64>()
        {
            Ok(MinLengthFilter::IncludeTotalPercentage(perc / 100.))
        } else if let Some(perc) = s.strip_suffix("%total")
            && let Ok(perc) = perc.parse::<f64>()
        {
            Ok(MinLengthFilter::PercentTotal(perc / 100.))
        } else {
            Err("Cannot parse".to_string())
        }
    }
}
