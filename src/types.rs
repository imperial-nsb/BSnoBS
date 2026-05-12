use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BubbleData {
    pub frame_name: String,
    pub bubble_id: usize,
    pub diameter_um: f32,
    pub volume_um3: f32,
    pub area_px: f32,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub is_valid: bool,
    pub is_static: bool,
}

#[derive(Clone, Debug)]
pub struct FrameResult {
    pub image_path: PathBuf,
    pub num_valid: usize,
    pub num_rejected: usize,
    pub bubbles: Vec<BubbleData>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnalysisParameters {
    pub scale_um_per_pixel: f32,
    #[serde(rename = "sample_volume_per_frame_uL")]
    pub sample_volume_per_frame_ul: f32,
    pub min_diameter_um: f32,
    pub max_diameter_um: f32,
    pub bin_size_um: f32,
}

impl Default for AnalysisParameters {
    fn default() -> Self {
        Self {
            scale_um_per_pixel: 0.0825,
            sample_volume_per_frame_ul: 0.00089,
            min_diameter_um: 0.5,
            max_diameter_um: 20.0,
            bin_size_um: 0.165,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnalysisResults {
    pub sample_name: String,
    pub parameters: AnalysisParameters,
    pub frames: Vec<FrameResult>,
}

impl AnalysisResults {
    pub fn total_bubbles(&self) -> usize {
        self.frames
            .iter()
            .flat_map(|f| f.bubbles.iter())
            .filter(|b| b.is_valid)
            .count()
    }

    pub fn total_rejected(&self) -> usize {
        self.frames.iter().map(|f| f.num_rejected).sum()
    }

    pub fn diameters_valid(&self) -> Vec<f32> {
        self.frames
            .iter()
            .flat_map(|f| f.bubbles.iter())
            .filter(|b| b.is_valid)
            .map(|b| b.diameter_um)
            .collect()
    }
}
