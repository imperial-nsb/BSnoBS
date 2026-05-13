//! CSV + JSON + PNG (overlay/histogram) exporter.

use std::path::Path;

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::{ImageReader, Rgb, RgbImage};
use imageproc::drawing::{
    draw_filled_rect_mut, draw_hollow_circle_mut, draw_line_segment_mut, draw_text_mut,
    text_size,
};
use imageproc::rect::Rect as IpRect;
use serde::Serialize;

use crate::types::{AnalysisResults, FrameResult};

#[derive(Clone, Copy, Debug)]
pub struct ExportOpts {
    pub csv: bool,
    pub png: bool,
    pub hist: bool,
    pub show_bubbles: bool,
    pub show_rejected: bool,
    pub show_static: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum HistAxisMode {
    Counts,
    AvgPerImage,
    Percent,
}

impl HistAxisMode {
    pub fn axis_label(self) -> &'static str {
        match self {
            HistAxisMode::Counts => "count",
            HistAxisMode::AvgPerImage => "avg / image",
            HistAxisMode::Percent => "%",
        }
    }
}

const FONT_BYTES: &[u8] = include_bytes!("../assets/Inter-Regular.ttf");

#[derive(Serialize)]
struct BubbleRow<'a> {
    frame: &'a str,
    bubble_id: usize,
    diameter_um: f32,
    volume_um3: f32,
    centroid_x_px: f32,
    centroid_y_px: f32,
}

pub fn export_csv_accepted(results: &AnalysisResults, out_dir: &Path) -> Result<()> {
    let sample = &results.sample_name;
    let mut wtr = csv::Writer::from_path(out_dir.join(format!("{sample}_bubble_data.csv")))?;
    for f in &results.frames {
        for b in &f.bubbles {
            if !b.is_valid {
                continue;
            }
            wtr.serialize(BubbleRow {
                frame: &b.frame_name,
                bubble_id: b.bubble_id,
                diameter_um: b.diameter_um,
                volume_um3: b.volume_um3,
                centroid_x_px: b.centroid_x,
                centroid_y_px: b.centroid_y,
            })?;
        }
    }
    wtr.flush()?;
    Ok(())
}

// ---------------- Overlay PNG ----------------

fn draw_thick_circle(img: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>, thickness: i32) {
    if r <= 0 {
        return;
    }
    let lo = (r - thickness / 2).max(1);
    let hi = r + (thickness - thickness / 2);
    for rr in lo..hi {
        draw_hollow_circle_mut(img, (cx, cy), rr, color);
    }
}

fn draw_dashed_circle(img: &mut RgbImage, cx: f32, cy: f32, r: f32, color: Rgb<u8>, thickness: i32) {
    let n = 24;
    for k in 0..n {
        if k % 2 == 1 {
            continue;
        }
        let a0 = (k as f32 / n as f32) * std::f32::consts::TAU;
        let a1 = ((k + 1) as f32 / n as f32) * std::f32::consts::TAU;
        let steps = 6;
        let mut prev = (cx + r * a0.cos(), cy + r * a0.sin());
        for s in 1..=steps {
            let t = s as f32 / steps as f32;
            let a = a0 + (a1 - a0) * t;
            let cur = (cx + r * a.cos(), cy + r * a.sin());
            for off in -(thickness / 2)..=(thickness - thickness / 2 - 1).max(0) {
                let dx = off as f32;
                draw_line_segment_mut(
                    img,
                    (prev.0 + dx, prev.1),
                    (cur.0 + dx, cur.1),
                    color,
                );
                draw_line_segment_mut(
                    img,
                    (prev.0, prev.1 + dx),
                    (cur.0, cur.1 + dx),
                    color,
                );
            }
            prev = cur;
        }
    }
}

pub fn render_overlay_png(
    frame: &FrameResult,
    um_per_pixel: f32,
    opts: ExportOpts,
    out_path: &Path,
) -> Result<()> {
    let src = ImageReader::open(&frame.image_path)
        .with_context(|| format!("opening {}", frame.image_path.display()))?
        .decode()?
        .to_rgb8();
    let mut img = src;

    let green = Rgb([0u8, 200, 0]);
    let red = Rgb([255u8, 23, 68]);
    let yellow = Rgb([255u8, 234, 0]);

    let font = FontRef::try_from_slice(FONT_BYTES).context("loading bundled font")?;
    let label_px = (img.height() as f32 / 80.0).clamp(10.0, 28.0);
    let label_scale = PxScale::from(label_px);

    let thickness = ((img.width().min(img.height()) as f32 / 600.0).round() as i32).clamp(2, 6);

    for b in &frame.bubbles {
        let draw_it = if b.is_static {
            opts.show_static
        } else if b.is_valid {
            opts.show_bubbles
        } else {
            opts.show_rejected
        };
        if !draw_it {
            continue;
        }
        let cx = b.centroid_x;
        let cy = b.centroid_y;
        let r = (b.diameter_um / 2.0) / um_per_pixel;
        if b.is_static {
            draw_dashed_circle(&mut img, cx, cy, r, yellow, thickness);
        } else {
            let color = if b.is_valid { green } else { red };
            draw_thick_circle(&mut img, cx as i32, cy as i32, r as i32, color, thickness);
            if b.is_valid {
                let text = format!("{:.1}", b.diameter_um);
                let (tw, th) = text_size(label_scale, &font, &text);
                let tx = (cx as i32) - (tw as i32) / 2;
                let ty = (cy as i32) - (th as i32) / 2;
                draw_text_mut(&mut img, color, tx, ty, label_scale, &font, &text);
            }
        }
    }

    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(out_path)?;
    Ok(())
}

// ---------------- Histogram PNG ----------------

const HIST_W: u32 = 900;
const HIST_H: u32 = 540;
const HIST_MARGIN_L: u32 = 70;
const HIST_MARGIN_R: u32 = 20;
const HIST_MARGIN_T: u32 = 40;
const HIST_MARGIN_B: u32 = 60;

fn bin_counts(diams: &[f32], min: f32, max: f32, n_bins: usize) -> Vec<u32> {
    let mut counts = vec![0u32; n_bins];
    if max <= min || n_bins == 0 {
        return counts;
    }
    let bw = (max - min) / n_bins as f32;
    for &d in diams {
        if d < min || d > max {
            continue;
        }
        let idx = (((d - min) / bw) as usize).min(n_bins - 1);
        counts[idx] += 1;
    }
    counts
}

fn scaled_value(c: u32, total: f32, n_frames: usize, mode: HistAxisMode) -> f32 {
    let c = c as f32;
    match mode {
        HistAxisMode::Counts => c,
        HistAxisMode::AvgPerImage => c / n_frames.max(1) as f32,
        HistAxisMode::Percent => {
            if total > 0.0 {
                c * 100.0 / total
            } else {
                0.0
            }
        }
    }
}

fn paint_histogram(
    img: &mut RgbImage,
    title: &str,
    diams: &[f32],
    n_frames: usize,
    x_min: f32,
    x_max: f32,
    y_max: f32,
    mode: HistAxisMode,
    bar_color: Rgb<u8>,
) {
    let font = FontRef::try_from_slice(FONT_BYTES).expect("font");
    let black = Rgb([20u8, 20, 20]);
    let grey = Rgb([170u8, 170, 170]);

    let plot_left = HIST_MARGIN_L as i32;
    let plot_right = (HIST_W - HIST_MARGIN_R) as i32;
    let plot_top = HIST_MARGIN_T as i32;
    let plot_bot = (HIST_H - HIST_MARGIN_B) as i32;
    let plot_w = plot_right - plot_left;
    let plot_h = plot_bot - plot_top;

    let title_scale = PxScale::from(20.0);
    let axis_scale = PxScale::from(14.0);

    draw_text_mut(img, black, plot_left, 10, title_scale, &font, title);

    if x_max <= x_min || y_max <= 0.0 {
        return;
    }

    let n_bins = 30;
    let counts = bin_counts(diams, x_min, x_max, n_bins);
    let total = diams.len() as f32;
    let bin_px = plot_w as f32 / n_bins as f32;

    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let v = scaled_value(c, total, n_frames, mode);
        let h = ((v / y_max).min(1.0) * plot_h as f32) as i32;
        if h <= 0 {
            continue;
        }
        let x0 = plot_left + (i as f32 * bin_px) as i32 + 1;
        let x1 = plot_left + ((i + 1) as f32 * bin_px) as i32 - 1;
        let w = (x1 - x0).max(1);
        let y0 = plot_bot - h;
        draw_filled_rect_mut(img, IpRect::at(x0, y0).of_size(w as u32, h as u32), bar_color);
    }

    draw_line_segment_mut(
        img,
        (plot_left as f32, plot_bot as f32),
        (plot_right as f32, plot_bot as f32),
        black,
    );
    draw_line_segment_mut(
        img,
        (plot_left as f32, plot_top as f32),
        (plot_left as f32, plot_bot as f32),
        black,
    );

    let n_ticks_x = 6;
    for k in 0..=n_ticks_x {
        let t = k as f32 / n_ticks_x as f32;
        let x = plot_left + (t * plot_w as f32) as i32;
        draw_line_segment_mut(
            img,
            (x as f32, plot_bot as f32),
            (x as f32, plot_bot as f32 + 4.0),
            black,
        );
        let v = x_min + t * (x_max - x_min);
        let label = format!("{:.1}", v);
        let (tw, _) = text_size(axis_scale, &font, &label);
        draw_text_mut(
            img,
            black,
            x - (tw as i32) / 2,
            plot_bot + 8,
            axis_scale,
            &font,
            &label,
        );
    }

    let n_ticks_y = 5;
    for k in 0..=n_ticks_y {
        let t = k as f32 / n_ticks_y as f32;
        let y = plot_bot - (t * plot_h as f32) as i32;
        draw_line_segment_mut(
            img,
            (plot_left as f32 - 4.0, y as f32),
            (plot_left as f32, y as f32),
            grey,
        );
        let v = t * y_max;
        let label = match mode {
            HistAxisMode::Counts => format!("{:.0}", v),
            HistAxisMode::AvgPerImage => {
                if y_max >= 10.0 {
                    format!("{:.0}", v)
                } else if y_max >= 1.0 {
                    format!("{:.1}", v)
                } else {
                    format!("{:.2}", v)
                }
            }
            HistAxisMode::Percent => format!("{:.0}%", v),
        };
        let (tw, th) = text_size(axis_scale, &font, &label);
        draw_text_mut(
            img,
            black,
            plot_left - 8 - tw as i32,
            y - (th as i32) / 2,
            axis_scale,
            &font,
            &label,
        );
    }

    let x_label = "diameter (μm)";
    let (tw, _) = text_size(axis_scale, &font, x_label);
    draw_text_mut(
        img,
        black,
        plot_left + (plot_w - tw as i32) / 2,
        plot_bot + 30,
        axis_scale,
        &font,
        x_label,
    );
    let y_label = mode.axis_label();
    draw_text_mut(img, black, 6, plot_top - 6, axis_scale, &font, y_label);
}

fn blank_canvas() -> RgbImage {
    let mut img = RgbImage::new(HIST_W, HIST_H);
    let white = Rgb([255u8, 255, 255]);
    draw_filled_rect_mut(&mut img, IpRect::at(0, 0).of_size(HIST_W, HIST_H), white);
    img
}

pub fn render_sample_histogram(
    results: &AnalysisResults,
    mode: HistAxisMode,
    x_min: f32,
    x_max: f32,
    y_max: f32,
    out_path: &Path,
) -> Result<()> {
    let diams = results.diameters_valid();
    let mut img = blank_canvas();
    paint_histogram(
        &mut img,
        &results.sample_name,
        &diams,
        results.frames.len(),
        x_min,
        x_max,
        y_max,
        mode,
        Rgb([60, 110, 200]),
    );
    img.save(out_path)?;
    Ok(())
}

pub fn render_combined_histogram(
    samples: &[(&str, &AnalysisResults)],
    mode: HistAxisMode,
    x_min: f32,
    x_max: f32,
    y_max: f32,
    out_path: &Path,
) -> Result<()> {
    if samples.is_empty() {
        return Ok(());
    }
    let cols = (samples.len() as f32).sqrt().ceil() as u32;
    let rows = (samples.len() as f32 / cols as f32).ceil() as u32;
    let w = cols * HIST_W;
    let h = rows * HIST_H;
    let mut canvas = RgbImage::new(w, h);
    draw_filled_rect_mut(&mut canvas, IpRect::at(0, 0).of_size(w, h), Rgb([255, 255, 255]));

    for (idx, (name, res)) in samples.iter().enumerate() {
        let mut sub = blank_canvas();
        paint_histogram(
            &mut sub,
            name,
            &res.diameters_valid(),
            res.frames.len(),
            x_min,
            x_max,
            y_max,
            mode,
            Rgb([60, 110, 200]),
        );
        let col = (idx as u32) % cols;
        let row = (idx as u32) / cols;
        let ox = col * HIST_W;
        let oy = row * HIST_H;
        for y in 0..HIST_H {
            for x in 0..HIST_W {
                canvas.put_pixel(ox + x, oy + y, *sub.get_pixel(x, y));
            }
        }
    }
    canvas.save(out_path)?;
    Ok(())
}

// ---------------- Helpers exposed for axis range ----------------

pub fn global_diameter_range(samples: &[&AnalysisResults]) -> (f32, f32) {
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for r in samples {
        for d in r.diameters_valid() {
            if d < mn {
                mn = d;
            }
            if d > mx {
                mx = d;
            }
        }
    }
    if mn.is_finite() && mx.is_finite() && mx > mn {
        (mn, mx)
    } else {
        (0.0, 1.0)
    }
}

pub fn global_y_max(samples: &[&AnalysisResults], min: f32, max: f32, mode: HistAxisMode) -> f32 {
    if matches!(mode, HistAxisMode::Percent) {
        return 100.0;
    }
    let n_bins = 30;
    let mut y_max: f32 = 0.0;
    for r in samples {
        let diams = r.diameters_valid();
        let counts = bin_counts(&diams, min, max, n_bins);
        let total = diams.len() as f32;
        let v = counts
            .iter()
            .map(|&c| scaled_value(c, total, r.frames.len(), mode))
            .fold(0.0_f32, f32::max);
        if v > y_max {
            y_max = v;
        }
    }
    if y_max > 0.0 {
        y_max
    } else {
        1.0
    }
}
