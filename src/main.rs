use anyhow::Result;
use eframe::egui;

mod app;
mod exporter;
mod inference;
mod static_filter;
mod types;

fn main() -> Result<()> {
    env_logger::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("BSnoBS — BubSize no BS (Rust)"),
        ..Default::default()
    };
    eframe::run_native(
        "bsnobs-rs",
        options,
        Box::new(|cc| Ok(Box::new(app::AppState::new(cc)))),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    Ok(())
}
