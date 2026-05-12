use anyhow::Result;
use eframe::egui;

mod app;
mod exporter;
mod inference;
mod static_filter;
mod types;

fn main() -> Result<()> {
    env_logger::init();

    let icon = {
        let bytes = include_bytes!("../assets/bs-logo.png");
        let img = image::load_from_memory(bytes)
            .expect("failed to decode embedded logo.png")
            .to_rgba8();
        let (w, h) = img.dimensions();
        egui::IconData {
            rgba: img.into_raw(),
            width: w,
            height: h,
        }
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("BSnoBS")
            .with_icon(icon),
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
