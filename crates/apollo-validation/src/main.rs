//! CLI entry point for Apollo validation.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = apollo_validation::run_full_suite()?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
