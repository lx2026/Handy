use anyhow::{anyhow, Context};
use ndarray::{s, Array1, Array2, Array3, Axis};
use ort::execution_providers::CPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use transcribe_rs::{TranscriptionResult, TranscriptionSegment};

#[derive(Debug, Clone)]
pub struct ParakeetCtcModelParams {
    pub prefer_int8: bool,
}

impl Default for ParakeetCtcModelParams {
    fn default() -> Self {
        Self { prefer_int8: true }
    }
}

impl ParakeetCtcModelParams {
    pub fn int8() -> Self {
        Self { prefer_int8: true }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ParakeetCtcInferenceParams;

#[derive(Debug, Clone)]
struct CtcFeatureConfig {
    sample_rate: u32,
    num_mel_bins: usize,
    n_fft: usize,
    window_size: f32,
    window_stride: f32,
    normalize_per_feature: bool,
}

impl Default for CtcFeatureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_mel_bins: 80,
            n_fft: 512,
            window_size: 0.025,
            window_stride: 0.01,
            normalize_per_feature: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ParakeetCtcConfigFile {
    sample_rate: Option<u32>,
    features: Option<usize>,
    num_mel_bins: Option<usize>,
    n_fft: Option<usize>,
    window_size: Option<f32>,
    window_stride: Option<f32>,
    normalize: Option<String>,
    blank_id: Option<usize>,
    model_file: Option<String>,
    vocab_file: Option<String>,
}

pub struct ParakeetCtcEngine {
    loaded_model_path: Option<PathBuf>,
    session: Option<Session>,
    vocab: Vec<String>,
    blank_id: usize,
    output_name: String,
    feature_config: CtcFeatureConfig,
}

impl Default for ParakeetCtcEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ParakeetCtcEngine {
    pub fn new() -> Self {
        Self {
            loaded_model_path: None,
            session: None,
            vocab: Vec::new(),
            blank_id: 0,
            output_name: "logprobs".to_string(),
            feature_config: CtcFeatureConfig::default(),
        }
    }

    pub fn load_model_with_params(
        &mut self,
        model_path: &Path,
        params: ParakeetCtcModelParams,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Self::load_config(model_path)?;
        let onnx_path = Self::resolve_model_file(model_path, cfg.model_file.as_deref(), params)?;
        let vocab_path = Self::resolve_vocab_file(model_path, cfg.vocab_file.as_deref())?;
        let vocab = Self::load_vocab(&vocab_path)?;
        if vocab.is_empty() {
            return Err(anyhow!("Vocabulary file is empty: {}", vocab_path.display()).into());
        }

        let feature_config = CtcFeatureConfig {
            sample_rate: cfg.sample_rate.unwrap_or(16_000),
            num_mel_bins: cfg.features.or(cfg.num_mel_bins).unwrap_or(80),
            n_fft: cfg.n_fft.unwrap_or(512),
            window_size: cfg.window_size.unwrap_or(0.025),
            window_stride: cfg.window_stride.unwrap_or(0.01),
            normalize_per_feature: cfg.normalize.as_deref().unwrap_or("per_feature")
                == "per_feature",
        };

        let providers = vec![CPUExecutionProvider::default().build()];
        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_execution_providers(providers)?
            .with_parallel_execution(true)?
            .commit_from_file(&onnx_path)?;

        let has_audio_input = session
            .inputs()
            .iter()
            .any(|input| input.name == "audio_signal");
        let has_length_input = session
            .inputs()
            .iter()
            .any(|input| input.name == "length");
        if !has_audio_input || !has_length_input {
            return Err(anyhow!(
                "CTC ONNX model must expose inputs 'audio_signal' and 'length': {}",
                onnx_path.display()
            )
            .into());
        }

        let output_name = if session
            .outputs()
            .iter()
            .any(|output| output.name == "logprobs")
        {
            "logprobs".to_string()
        } else {
            session
                .outputs()
                .first()
                .map(|output| output.name.to_string())
                .ok_or_else(|| anyhow!("Model has no outputs: {}", onnx_path.display()))?
        };

        self.loaded_model_path = Some(model_path.to_path_buf());
        self.session = Some(session);
        self.vocab = vocab;
        self.blank_id = cfg.blank_id.unwrap_or(self.vocab.len()).min(self.vocab.len());
        self.output_name = output_name;
        self.feature_config = feature_config;
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.loaded_model_path = None;
        self.session = None;
        self.vocab.clear();
    }

    pub fn transcribe_samples(
        &mut self,
        samples: Vec<f32>,
        _params: Option<ParakeetCtcInferenceParams>,
    ) -> Result<TranscriptionResult, Box<dyn std::error::Error>> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("Parakeet CTC model not loaded"))?;

        if samples.is_empty() {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: Some(vec![]),
            });
        }

        let features = compute_log_mel_features(&samples, &self.feature_config);
        if features.nrows() == 0 {
            return Ok(TranscriptionResult {
                text: String::new(),
                segments: Some(vec![]),
            });
        }

        // CTC ONNX input shape: [B, 80, T]
        let audio_signal = features.t().to_owned().insert_axis(Axis(0));
        let length = Array1::from_vec(vec![features.nrows() as i64]);

        let inputs = inputs![
            "audio_signal" => TensorRef::from_array_view(audio_signal.view())?,
            "length" => TensorRef::from_array_view(length.view())?,
        ];
        let outputs = session.run(inputs)?;

        let logprobs = outputs
            .get(self.output_name.as_str())
            .ok_or_else(|| anyhow!("Model output '{}' not found", self.output_name))?
            .try_extract_array::<f32>()?
            .to_owned()
            .into_dimensionality::<ndarray::Ix3>()?;

        let token_ids = ctc_greedy_decode(&logprobs, self.blank_id);
        let text = decode_token_ids(&token_ids, &self.vocab);
        let duration = samples.len() as f32 / self.feature_config.sample_rate as f32;

        Ok(TranscriptionResult {
            text: text.clone(),
            segments: Some(vec![TranscriptionSegment {
                start: 0.0,
                end: duration.max(0.0),
                text,
            }]),
        })
    }

    fn load_config(model_dir: &Path) -> Result<ParakeetCtcConfigFile, Box<dyn std::error::Error>> {
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Ok(ParakeetCtcConfigFile::default());
        }

        let text = fs::read_to_string(&config_path)
            .with_context(|| format!("Failed reading {}", config_path.display()))?;
        let cfg: ParakeetCtcConfigFile = serde_json::from_str(&text)
            .with_context(|| format!("Failed parsing {}", config_path.display()))?;
        Ok(cfg)
    }

    fn resolve_model_file(
        model_dir: &Path,
        configured_file: Option<&str>,
        params: ParakeetCtcModelParams,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(file) = configured_file {
            let candidate = model_dir.join(file);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        let defaults = [
            "model.onnx",
            "parakeet_ctc_0p6b_zh_cn_int8.onnx",
            "parakeet_ctc_0p6b_zh_cn_fp32.onnx",
        ];
        for file in defaults {
            let candidate = model_dir.join(file);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        let mut candidates = Vec::new();
        for entry in fs::read_dir(model_dir)? {
            let path = entry?.path();
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                candidates.push(path);
            }
        }
        if candidates.is_empty() {
            return Err(anyhow!("No .onnx files found in {}", model_dir.display()).into());
        }
        candidates.sort();

        if params.prefer_int8 {
            if let Some(path) = candidates.iter().find(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.contains("int8"))
                    .unwrap_or(false)
            }) {
                return Ok(path.clone());
            }
        }

        Ok(candidates[0].clone())
    }

    fn resolve_vocab_file(
        model_dir: &Path,
        configured_file: Option<&str>,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(file) = configured_file {
            let candidate = model_dir.join(file);
            if candidate.exists() {
                return Ok(candidate);
            }
        }

        let mut candidates = Vec::new();
        for entry in fs::read_dir(model_dir)? {
            let path = entry?.path();
            if !path.is_file() {
                continue;
            }
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if name == "vocab.txt" || name.ends_with("vocab.txt") {
                candidates.push(path);
            }
        }

        if candidates.is_empty() {
            return Err(anyhow!("No vocab file found in {}", model_dir.display()).into());
        }
        candidates.sort();
        Ok(candidates[0].clone())
    }

    fn load_vocab(path: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Failed reading {}", path.display()))?;
        let mut indexed = Vec::<(usize, String)>::new();
        let mut plain = Vec::<String>::new();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Handle sentencepiece style vocab: "<token>\t<score>"
            if let Some((token, score_text)) = line.split_once('\t') {
                if score_text.trim().parse::<f32>().is_ok() {
                    plain.push(token.to_string());
                    continue;
                }
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.is_empty() {
                continue;
            }

            // Handle indexed vocab style: "<id> <token>"
            if parts.len() >= 2 {
                if let Ok(id) = parts[0].parse::<usize>() {
                    indexed.push((id, parts[1].to_string()));
                    continue;
                }
            }

            // Handle indexed vocab style: "<token> <id>"
            if parts.len() >= 2 {
                if let Ok(id) = parts[parts.len() - 1].parse::<usize>() {
                    indexed.push((id, parts[..parts.len() - 1].join(" ")));
                    continue;
                }
            }

            // Handle score-formatted vocab: "<token> <score>"
            if parts.len() >= 2 && parts[parts.len() - 1].parse::<f32>().is_ok() {
                plain.push(parts[..parts.len() - 1].join(" "));
                continue;
            }

            plain.push(line.to_string());
        }

        if !indexed.is_empty() {
            let max_id = indexed.iter().map(|(id, _)| *id).max().unwrap_or(0);
            let mut vocab = vec![String::new(); max_id + 1];
            for (id, token) in indexed {
                vocab[id] = token;
            }
            Ok(vocab)
        } else {
            Ok(plain)
        }
    }
}

fn ctc_greedy_decode(logprobs: &Array3<f32>, blank_id: usize) -> Vec<usize> {
    let shape = logprobs.shape();
    if shape.len() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return vec![];
    }

    let mut ids = Vec::new();
    let mut prev = usize::MAX;
    for timestep in 0..shape[1] {
        let row = logprobs.slice(s![0, timestep, ..]);
        let mut best_id = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for (token_id, score) in row.iter().enumerate() {
            if *score > best_score {
                best_score = *score;
                best_id = token_id;
            }
        }
        if best_id != blank_id && best_id != prev {
            ids.push(best_id);
        }
        prev = best_id;
    }
    ids
}

fn decode_token_ids(token_ids: &[usize], vocab: &[String]) -> String {
    let mut text = String::new();
    for token_id in token_ids {
        let Some(token) = vocab.get(*token_id) else {
            continue;
        };
        if token.is_empty()
            || token == "<blk>"
            || token == "<blank>"
            || token == "<unk>"
            || token == "<s>"
            || token == "</s>"
            || token == "<pad>"
        {
            continue;
        }

        let mut piece = token.trim();
        if piece.is_empty() {
            continue;
        }

        if piece.starts_with("##") {
            text.push_str(piece.trim_start_matches("##"));
            continue;
        }

        // SentencePiece word-boundary marker.
        let mut starts_new_word = false;
        if piece.starts_with('▁') {
            starts_new_word = true;
            piece = piece.trim_start_matches('▁');
            if piece.is_empty() {
                continue;
            }
        }

        if is_cjk_token(piece) || is_punctuation_token(piece) {
            while text.ends_with(' ') {
                text.pop();
            }
            text.push_str(piece);
        } else if text.is_empty() {
            text.push_str(piece);
        } else {
            if starts_new_word || !text.ends_with(' ') {
                text.push(' ');
            }
            text.push_str(piece);
        }
    }

    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn is_punctuation_token(token: &str) -> bool {
    token.chars().all(|c| {
        c.is_ascii_punctuation()
            || matches!(
                c,
                '，' | '。' | '！' | '？' | '：' | '；' | '、' | '（' | '）' | '“' | '”'
            )
    })
}

fn is_cjk_token(token: &str) -> bool {
    !token.is_empty()
        && token.chars().all(|c| {
            matches!(
                c as u32,
                0x3400..=0x4DBF
                    | 0x4E00..=0x9FFF
                    | 0x20000..=0x2A6DF
                    | 0x2A700..=0x2B73F
                    | 0x2B740..=0x2B81F
                    | 0x2B820..=0x2CEAF
            )
        })
}

fn compute_log_mel_features(samples: &[f32], config: &CtcFeatureConfig) -> Array2<f32> {
    let frame_length = (config.window_size * config.sample_rate as f32) as usize;
    let frame_shift = (config.window_stride * config.sample_rate as f32) as usize;
    if frame_length == 0 || frame_shift == 0 {
        return Array2::zeros((0, config.num_mel_bins));
    }

    let num_frames = if samples.len() < frame_length {
        1
    } else {
        1 + (samples.len() - frame_length) / frame_shift
    };

    let fft_size = config.n_fft.max(frame_length.next_power_of_two());
    let fft_bins = fft_size / 2 + 1;
    let window = hann_window(frame_length);
    let mel_banks = mel_filterbank(
        config.num_mel_bins,
        fft_size,
        config.sample_rate as f32,
        0.0,
        config.sample_rate as f32 / 2.0,
    );

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut features = Array2::<f32>::zeros((num_frames, config.num_mel_bins));

    for frame_idx in 0..num_frames {
        let start = frame_idx * frame_shift;
        let mut frame = vec![0.0f32; frame_length];
        let copy_len = frame_length.min(samples.len().saturating_sub(start));
        frame[..copy_len].copy_from_slice(&samples[start..start + copy_len]);

        for (i, sample) in frame.iter_mut().enumerate() {
            *sample *= window[i];
        }

        let mut fft_input: Vec<Complex<f32>> =
            frame.iter().map(|&value| Complex::new(value, 0.0)).collect();
        fft_input.resize(fft_size, Complex::new(0.0, 0.0));
        fft.process(&mut fft_input);

        let power: Vec<f32> = fft_input[..fft_bins].iter().map(|value| value.norm_sqr()).collect();
        for mel_idx in 0..config.num_mel_bins {
            let mut mel_energy = 0.0f32;
            for (weight, spectrum) in mel_banks.row(mel_idx).iter().zip(power.iter()) {
                mel_energy += weight * spectrum;
            }
            features[[frame_idx, mel_idx]] = mel_energy.max(1.0e-10).ln();
        }
    }

    if config.normalize_per_feature && num_frames > 1 {
        for mel_idx in 0..config.num_mel_bins {
            let col = features.column(mel_idx);
            let mean = col.iter().sum::<f32>() / num_frames as f32;
            let var = col
                .iter()
                .map(|value| {
                    let centered = *value - mean;
                    centered * centered
                })
                .sum::<f32>()
                / num_frames as f32;
            let std = (var + 1.0e-5).sqrt();
            for frame_idx in 0..num_frames {
                features[[frame_idx, mel_idx]] = (features[[frame_idx, mel_idx]] - mean) / std;
            }
        }
    }

    features
}

fn hann_window(length: usize) -> Vec<f32> {
    use std::f32::consts::PI;

    if length <= 1 {
        return vec![1.0; length];
    }
    (0..length)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / (length as f32 - 1.0)).cos())
        .collect()
}

fn mel_filterbank(
    num_mel_bins: usize,
    n_fft: usize,
    sample_rate: f32,
    low_freq: f32,
    high_freq: f32,
) -> Array2<f32> {
    let fft_bins = n_fft / 2 + 1;
    let mel_low = hz_to_mel(low_freq);
    let mel_high = hz_to_mel(high_freq);
    let num_points = num_mel_bins + 2;

    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.into_iter().map(mel_to_hz).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|freq| freq * n_fft as f32 / sample_rate)
        .collect();

    let mut banks = Array2::<f32>::zeros((num_mel_bins, fft_bins));
    for mel_idx in 0..num_mel_bins {
        let left = bin_points[mel_idx];
        let center = bin_points[mel_idx + 1];
        let right = bin_points[mel_idx + 2];

        for bin_idx in 0..fft_bins {
            let bin = bin_idx as f32;
            let value = if bin > left && bin < center {
                (bin - left) / (center - left)
            } else if bin >= center && bin < right {
                (right - bin) / (right - center)
            } else {
                0.0
            };
            banks[[mel_idx, bin_idx]] = value.max(0.0);
        }
    }
    banks
}

fn hz_to_mel(hz: f32) -> f32 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}
