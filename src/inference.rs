//! ONNX inference with letterbox preprocess + class-agnostic NMS.
//!
//! Mirrors the math used by `bsnobs.infer.StudentAnalyzer`.

use std::path::Path;

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView, ImageReader};
use ndarray::{Array4, ArrayView2, Axis};
use ort::{
    execution_providers::{CPUExecutionProvider, CoreMLExecutionProvider},
    session::Session,
    value::Value,
};

use crate::types::{AnalysisParameters, BubbleData, FrameResult};

pub const INPUT_SIZE: u32 = 640;

/// Bundled student weights, embedded in the binary at compile time.
pub const BUNDLED_WEIGHTS: &[u8] = include_bytes!("../assets/student.onnx");

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Device {
    Auto,
    Cpu,
    CoreML,
}

impl Device {
    pub fn label(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::CoreML => "coreml",
        }
    }
}

pub struct StudentAnalyzer {
    session: Session,
}

impl StudentAnalyzer {
    fn build(device: Device) -> Result<ort::session::builder::SessionBuilder> {
        let mut builder = Session::builder()?;
        let providers = match device {
            Device::Cpu => vec![CPUExecutionProvider::default().build()],
            Device::CoreML | Device::Auto => vec![
                CoreMLExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ],
        };
        builder = builder.with_execution_providers(providers)?;
        Ok(builder)
    }

    pub fn load(weights: &Path, device: Device) -> Result<Self> {
        let session = Self::build(device)?
            .commit_from_file(weights)
            .with_context(|| format!("loading ONNX model from {}", weights.display()))?;
        Ok(Self { session })
    }

    pub fn load_from_bytes(bytes: &[u8], device: Device) -> Result<Self> {
        let session = Self::build(device)?
            .commit_from_memory(bytes)
            .context("loading ONNX model from embedded bytes")?;
        Ok(Self { session })
    }

    pub fn predict_image(
        &mut self,
        image_path: &Path,
        params: &AnalysisParameters,
        conf_thr: f32,
        iou_thr: f32,
        max_det: usize,
    ) -> Result<FrameResult> {
        let img = ImageReader::open(image_path)?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("decoding image {}", image_path.display()))?;
        let (tensor, pad_x, pad_y, scale) = letterbox(&img, INPUT_SIZE);

        let value = Value::from_array(tensor)?;
        let outputs = self.session.run(ort::inputs!["images" => value])?;
        let (_, out) = outputs
            .iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("model produced no outputs"))?;
        let arr = out.try_extract_array::<f32>()?;
        let view3 = arr
            .view()
            .into_dimensionality::<ndarray::Ix3>()?;
        // [1, 5, N] -> [5, N]
        let preds = view3.index_axis(Axis(0), 0);
        let detections = decode_and_nms(preds, conf_thr, iou_thr, max_det);

        let frame_name = image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("frame")
            .to_string();

        let mut bubbles = Vec::with_capacity(detections.len());
        let mut num_valid = 0usize;
        let mut num_rejected = 0usize;
        for (i, d) in detections.iter().enumerate() {
            let cx_src = (d.cx - pad_x) / scale;
            let cy_src = (d.cy - pad_y) / scale;
            let w_src = d.w / scale;
            let h_src = d.h / scale;

            let r_px = (w_src.min(h_src) + (w_src * h_src).sqrt()) / 4.0;
            let area_px = std::f32::consts::PI * r_px * r_px;
            let diameter_um = 2.0 * r_px * params.scale_um_per_pixel;
            let radius_um = diameter_um / 2.0;
            let volume_um3 = 4.0 / 3.0 * std::f32::consts::PI * radius_um.powi(3);

            let is_valid = diameter_um >= params.min_diameter_um
                && diameter_um <= params.max_diameter_um;
            if is_valid {
                num_valid += 1;
            } else {
                num_rejected += 1;
            }
            bubbles.push(BubbleData {
                frame_name: frame_name.clone(),
                bubble_id: i + 1,
                diameter_um,
                volume_um3,
                area_px,
                centroid_x: cx_src,
                centroid_y: cy_src,
                is_valid,
                is_static: false,
            });
        }

        Ok(FrameResult {
            image_path: image_path.to_path_buf(),
            num_valid,
            num_rejected,
            bubbles,
        })
    }
}

/// Letterbox the image into a [1,3,size,size] f32 NCHW tensor (normalized 0..1, RGB).
/// Returns (tensor, pad_x, pad_y, scale) — to undo letterbox: src = (xy - pad) / scale.
fn letterbox(img: &DynamicImage, size: u32) -> (Array4<f32>, f32, f32, f32) {
    let (w, h) = img.dimensions();
    let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
    let new_w = (w as f32 * scale).round() as u32;
    let new_h = (h as f32 * scale).round() as u32;
    let pad_x = (size as f32 - new_w as f32) / 2.0;
    let pad_y = (size as f32 - new_h as f32) / 2.0;

    let resized = img
        .resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
        .to_rgb8();

    let mut tensor =
        Array4::<f32>::from_elem((1, 3, size as usize, size as usize), 114.0 / 255.0);
    for y in 0..new_h {
        for x in 0..new_w {
            let p = resized.get_pixel(x, y);
            let tx = (pad_x as u32 + x) as usize;
            let ty = (pad_y as u32 + y) as usize;
            tensor[[0, 0, ty, tx]] = p[0] as f32 / 255.0;
            tensor[[0, 1, ty, tx]] = p[1] as f32 / 255.0;
            tensor[[0, 2, ty, tx]] = p[2] as f32 / 255.0;
        }
    }
    (tensor, pad_x, pad_y, scale)
}

#[derive(Clone, Debug)]
struct Det {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    conf: f32,
}

fn decode_and_nms(preds: ArrayView2<f32>, conf_thr: f32, iou_thr: f32, max_det: usize) -> Vec<Det> {
    let n = preds.shape()[1];
    let mut raw: Vec<Det> = Vec::new();
    for i in 0..n {
        let conf = preds[[4, i]];
        if conf >= conf_thr {
            raw.push(Det {
                cx: preds[[0, i]],
                cy: preds[[1, i]],
                w: preds[[2, i]],
                h: preds[[3, i]],
                conf,
            });
        }
    }
    raw.sort_by(|a, b| b.conf.partial_cmp(&a.conf).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<Det> = Vec::new();
    for d in raw {
        if kept.len() >= max_det {
            break;
        }
        if kept.iter().all(|k| iou(&d, k) <= iou_thr) {
            kept.push(d);
        }
    }
    kept
}

fn iou(a: &Det, b: &Det) -> f32 {
    let (ax1, ay1) = (a.cx - a.w / 2.0, a.cy - a.h / 2.0);
    let (ax2, ay2) = (a.cx + a.w / 2.0, a.cy + a.h / 2.0);
    let (bx1, by1) = (b.cx - b.w / 2.0, b.cy - b.h / 2.0);
    let (bx2, by2) = (b.cx + b.w / 2.0, b.cy + b.h / 2.0);
    let iw = (ax2.min(bx2) - ax1.max(bx1)).max(0.0);
    let ih = (ay2.min(by2) - ay1.max(by1)).max(0.0);
    let inter = iw * ih;
    let area_a = (ax2 - ax1).max(0.0) * (ay2 - ay1).max(0.0);
    let area_b = (bx2 - bx1).max(0.0) * (by2 - by1).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

