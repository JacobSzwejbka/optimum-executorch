#!/usr/bin/env python3
"""Export nvidia/parakeet-tdt-0.6b-v3 components to ExecuTorch."""

import os

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


# ============================================================================
# Audio Preprocessing (Mel Spectrogram)
# ============================================================================


class MelSpectrogramPreprocessor(torch.nn.Module):
    """Mel spectrogram preprocessor matching NeMo's FilterbankFeatures."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_window_size: int = 400,  # 25ms at 16kHz
        n_window_stride: int = 160,  # 10ms at 16kHz
        n_fft: int = 512,
        nfilt: int = 80,
        preemph: float = 0.97,
        log_zero_guard_value: float = 1e-10,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_window_size = n_window_size
        self.n_window_stride = n_window_stride
        self.n_fft = n_fft
        self.nfilt = nfilt
        self.preemph = preemph
        self.log_zero_guard_value = log_zero_guard_value

        # Create mel filterbank
        import librosa

        filterbanks = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=nfilt,
            fmin=0.0,
            fmax=sample_rate / 2,
        )
        self.register_buffer("fb", torch.tensor(filterbanks, dtype=torch.float32))

        # Hann window
        self.register_buffer("window", torch.hann_window(n_window_size))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert raw audio waveform to log mel spectrogram.

        Args:
            audio: [batch, samples] raw audio at 16kHz

        Returns:
            mel: [batch, n_mels, time] log mel spectrogram
            length: [batch] number of valid frames
        """
        batch_size = audio.shape[0]

        # Pre-emphasis: y[t] = x[t] - 0.97 * x[t-1]
        audio = torch.cat(
            [audio[:, :1], audio[:, 1:] - self.preemph * audio[:, :-1]],
            dim=1,
        )

        # STFT
        x = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.n_window_stride,
            win_length=self.n_window_size,
            window=self.window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )

        # Magnitude -> Power spectrum
        x = x.abs().pow(2)  # [batch, n_fft//2+1, time]

        # Apply mel filterbank
        x = torch.matmul(self.fb, x)  # [batch, n_mels, time]

        # Log compression
        x = torch.log(x + self.log_zero_guard_value)

        # Compute valid lengths
        num_samples = audio.shape[1]
        length = torch.tensor(
            [(num_samples + self.n_window_stride - 1) // self.n_window_stride] * batch_size,
            dtype=torch.int64,
        )

        return x, length


def load_audio(audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
    """Load audio file and resample to target sample rate."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
    except (ImportError, Exception):
        # Fallback to scipy
        from scipy.io import wavfile

        sr, data = wavfile.read(audio_path)
        # Convert to float32 and normalize
        if data.dtype == "int16":
            data = data.astype("float32") / 32768.0
        elif data.dtype == "int32":
            data = data.astype("float32") / 2147483648.0
        waveform = torch.from_numpy(data).unsqueeze(0)  # [1, samples]

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != sample_rate:
        try:
            import torchaudio

            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        except ImportError:
            # Simple resampling with scipy
            from scipy import signal

            num_samples = int(len(waveform[0]) * sample_rate / sr)
            resampled = signal.resample(waveform[0].numpy(), num_samples)
            waveform = torch.from_numpy(resampled).unsqueeze(0).float()

    return waveform  # [1, samples]


# ============================================================================
# Greedy Decoding
# ============================================================================


def greedy_decode_eager(encoder_output: torch.Tensor, encoder_len: torch.Tensor, model) -> list[int]:
    """
    Greedy decode using NeMo's built-in decoding.

    Args:
        encoder_output: [1, hidden, time] from encoder
        encoder_len: [1] tensor with number of valid encoder frames
        model: NeMo ASR model

    Returns:
        List of token IDs
    """
    # Use NeMo's built-in decoding which handles all RNN-T complexity
    hypotheses = model.decoding.rnnt_decoder_predictions_tensor(
        encoder_output=encoder_output,
        encoded_lengths=encoder_len,
        return_hypotheses=True,
    )
    # hypotheses is a list of Hypothesis objects
    return hypotheses[0].y_sequence


class DecoderSOS(torch.nn.Module):
    """Returns zeros for start-of-sequence (SOS)."""

    def __init__(self, decoder):
        super().__init__()
        self.pred_hidden = decoder.pred_hidden

    def forward(self) -> torch.Tensor:
        """Return zeros for SOS."""
        return torch.zeros(1, 1, self.pred_hidden)


class DecoderPredict(torch.nn.Module):
    """Wrapper for decoder.predict() for actual tokens with LSTM state."""

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        self.pred_hidden = decoder.pred_hidden
        self.pred_rnn_layers = getattr(decoder, "pred_rnn_layers", 2)

    def forward(
        self, token: torch.Tensor, h: torch.Tensor, c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for decoder prediction with explicit LSTM state.

        Args:
            token: [1, 1] int64 tensor with token ID
            h: [num_layers, 1, hidden] LSTM hidden state
            c: [num_layers, 1, hidden] LSTM cell state

        Returns:
            g: [1, 1, hidden] decoder output
            h_new: [num_layers, 1, hidden] new hidden state
            c_new: [num_layers, 1, hidden] new cell state
        """
        state = [h, c]
        g, new_state = self.decoder.predict(y=token, state=state, add_sos=False)
        return g, new_state[0], new_state[1]


def greedy_decode_executorch(
    encoder_output: torch.Tensor,
    encoder_len: int,
    program,
    blank_id: int,
    vocab_size: int,
    num_rnn_layers: int = 2,
    pred_hidden: int = 640,
    max_symbols_per_step: int = 10,
    durations: list[int] | None = None,
    debug: bool = False,
) -> list[int]:
    """
    TDT duration-aware greedy decode using ExecuTorch runtime with proper LSTM state.

    This implements NeMo's label-looping TDT algorithm:
    - Each iteration emits at most one non-blank token
    - After emitting non-blank, decoder state is updated before next prediction
    - Duration determines how many frames to advance (0 = stay on same frame)
    - For blank predictions, duration is forced to >= 1

    Args:
        encoder_output: [1, hidden, time] from encoder
        encoder_len: number of valid encoder frames
        program: ExecuTorch program with loaded methods
        blank_id: blank token ID (vocab_size for TDT)
        vocab_size: vocabulary size (8192)
        num_rnn_layers: number of LSTM layers (2 for parakeet-tdt)
        pred_hidden: decoder hidden dimension (640 for parakeet-tdt)
        max_symbols_per_step: max tokens per encoder frame (safety limit)
        durations: list of possible duration values (default: [0, 1, 2, 3, 4])
        debug: if True, print debug info

    Returns:
        List of token IDs
    """
    if durations is None:
        durations = [0, 1, 2, 3, 4]  # TDT default durations

    hypothesis = []
    num_token_classes = vocab_size + 1  # vocab + blank (8193)

    # Transpose encoder output to [1, time, hidden]
    encoder_output = encoder_output.transpose(1, 2)

    # Project encoder output once (ensure contiguous)
    proj_enc_method = program.load_method("joint_project_encoder")
    f_proj = proj_enc_method.execute([encoder_output.contiguous()])[0]  # [1, T, joint_hidden]

    # Load other methods
    decoder_sos_method = program.load_method("decoder_sos")
    decoder_predict_method = program.load_method("decoder_predict")
    proj_dec_method = program.load_method("joint_project_decoder")
    joint_method = program.load_method("joint")

    # Initialize LSTM state (zeros)
    h = torch.zeros(num_rnn_layers, 1, pred_hidden)
    c = torch.zeros(num_rnn_layers, 1, pred_hidden)

    # Get SOS output and project it
    sos_g = decoder_sos_method.execute([])[0]  # [1, 1, hidden]
    g_proj = proj_dec_method.execute([sos_g])[0]  # [1, 1, joint_hidden]

    t = 0  # Current time index
    symbols_on_frame = 0  # Track symbols emitted on current frame for safety

    while t < encoder_len:
        f_t = f_proj[:, t : t + 1, :].contiguous()  # [1, 1, joint_hidden]

        # Joint forward - TDT returns concatenated [token_logits, duration_logits]
        joint_out = joint_method.execute([f_t, g_proj])

        # Full output: [8198] = [8193 tokens + 5 durations]
        full_logits = joint_out[0].squeeze()
        token_logits = full_logits[:num_token_classes]  # [8193]
        duration_logits = full_logits[num_token_classes:]  # [5]

        # Argmax for token and duration
        k = token_logits.argmax().item()
        dur_idx = duration_logits.argmax().item()
        dur = durations[dur_idx]

        if debug and len(hypothesis) < 20:
            print(f"    t={t}, k={k}, dur={dur}, blank={k == blank_id}")

        if k == blank_id:
            # Blank predicted - advance by duration (at least 1)
            dur = max(dur, 1)
            t += dur
            symbols_on_frame = 0
        else:
            # Non-blank token - add to hypothesis
            hypothesis.append(k)

            # Update decoder state with this token (passing and receiving LSTM state)
            token = torch.tensor([[k]], dtype=torch.long)
            result = decoder_predict_method.execute([token, h, c])
            g = result[0]  # [1, 1, hidden]
            h = result[1]  # [num_layers, 1, hidden]
            c = result[2]  # [num_layers, 1, hidden]

            # Project decoder output
            g_proj = proj_dec_method.execute([g])[0]  # [1, 1, joint_hidden]

            # Advance time by duration
            t += dur

            # Safety: track symbols per frame to avoid infinite loops
            if dur == 0:
                symbols_on_frame += 1
                if symbols_on_frame >= max_symbols_per_step:
                    t += 1  # Force advance
                    symbols_on_frame = 0
            else:
                symbols_on_frame = 0

    return hypothesis


# ============================================================================
# End-to-End Inference
# ============================================================================


def transcribe_eager(audio_path: str, model) -> str:
    """
    Transcribe audio file using eager PyTorch model.

    Args:
        audio_path: path to .wav file
        model: NeMo ASR model

    Returns:
        Transcribed text
    """
    model.eval()

    with torch.no_grad():
        # Load audio
        audio = load_audio(audio_path)  # [1, samples]
        print(f"  Loaded audio: {audio.shape[1]} samples ({audio.shape[1] / 16000:.2f}s)")

        # Preprocess to mel spectrogram using model's preprocessor
        mel, mel_len = model.preprocessor(input_signal=audio, length=torch.tensor([audio.shape[1]]))
        print(f"  Mel spectrogram: {mel.shape}")

        # Encode
        encoded, encoded_len = model.encoder(audio_signal=mel, length=mel_len)
        print(f"  Encoder output: {encoded.shape}, len={encoded_len.item()}")

        # Greedy decode using NeMo's built-in decoding
        tokens = greedy_decode_eager(encoded, encoded_len, model)
        print(f"  Decoded {len(tokens)} tokens")

        # Convert tokens to text
        text = model.tokenizer.ids_to_text(tokens)

    return text


def transcribe_executorch(audio_path: str, model, et_buffer) -> str:
    """
    Transcribe audio file using ExecuTorch runtime.

    Args:
        audio_path: path to .wav file
        model: NeMo ASR model (for tokenizer and preprocessor)
        et_buffer: ExecuTorch program buffer

    Returns:
        Transcribed text
    """
    from executorch.runtime import Runtime

    runtime = Runtime.get()
    program = runtime.load_program(et_buffer)

    with torch.no_grad():
        # Load audio
        audio = load_audio(audio_path)  # [1, samples]
        print(f"  Loaded audio: {audio.shape[1]} samples ({audio.shape[1] / 16000:.2f}s)")

        # Preprocess to mel spectrogram using model's preprocessor
        # (preprocessor not exported, use eager model)
        mel, mel_len = model.preprocessor(input_signal=audio, length=torch.tensor([audio.shape[1]]))
        print(f"  Mel spectrogram: {mel.shape}")

        # Encode using ExecuTorch
        encoder_method = program.load_method("encoder")
        enc_result = encoder_method.execute([mel, mel_len])
        encoded = enc_result[0]
        encoded_len = enc_result[1].item()
        print(f"  Encoder output: {encoded.shape}, len={encoded_len}")

        # Greedy decode using ExecuTorch
        # For TDT, blank_id is vocab_size (not num_classes_with_blank which includes durations)
        vocab_size = model.tokenizer.vocab_size  # 8192
        blank_id = vocab_size  # blank is at index 8192
        tokens = greedy_decode_executorch(
            encoded,
            encoded_len,
            program,
            blank_id,
            vocab_size,
            num_rnn_layers=model.decoder.pred_rnn_layers,
            pred_hidden=model.decoder.pred_hidden,
        )
        print(f"  Decoded {len(tokens)} tokens")

        # Convert tokens to text
        text = model.tokenizer.ids_to_text(tokens)

    return text


def load_model():
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    model.eval()
    return model


class JointAfterProjection(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, f, g):
        return self.joint.joint_after_projection(f, g)


class JointProjectEncoder(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, f):
        return self.joint.project_encoder(f)


class JointProjectDecoder(torch.nn.Module):
    def __init__(self, joint):
        super().__init__()
        self.joint = joint

    def forward(self, g):
        return self.joint.project_prednet(g)


def export_all(model):
    """Export all components, return dict of ExportedPrograms."""
    programs = {}

    # Encoder - use Dim.AUTO for Conformer subsampling constraints
    feat_in = getattr(model.encoder, "_feat_in", 80)
    audio_signal = torch.randn(1, feat_in, 100)
    length = torch.tensor([100], dtype=torch.int64)
    programs["encoder"] = export(
        model.encoder,
        (),
        kwargs={"audio_signal": audio_signal, "length": length},
        dynamic_shapes={
            "audio_signal": {2: Dim.AUTO},
            "length": {},
        },
        strict=False,
    )
    print("Exported encoder")

    # Decoder - batch mode (no streaming state needed)
    # For 30s chunks, process with states=None (internal state handling)
    targets = torch.zeros(1, 1, dtype=torch.long)
    target_length = torch.tensor([1], dtype=torch.int64)
    programs["decoder"] = export(
        model.decoder,
        (),
        kwargs={"targets": targets, "target_length": target_length, "states": None},
        dynamic_shapes={"targets": {}, "target_length": {}, "states": None},
        strict=False,
    )
    print("Exported decoder (batch mode)")

    # Decoder predict wrapper - for greedy decode loop with LSTM state
    decoder_predict = DecoderPredict(model.decoder)
    decoder_predict.eval()
    token = torch.tensor([[0]], dtype=torch.long)  # dummy token
    num_layers = model.decoder.pred_rnn_layers
    pred_hidden = model.decoder.pred_hidden
    h = torch.zeros(num_layers, 1, pred_hidden)  # LSTM hidden state
    c = torch.zeros(num_layers, 1, pred_hidden)  # LSTM cell state
    programs["decoder_predict"] = export(
        decoder_predict,
        (token, h, c),
        dynamic_shapes={"token": {}, "h": {}, "c": {}},
        strict=False,
    )
    print("Exported decoder_predict")

    # Decoder SOS - returns zeros for start-of-sequence
    decoder_sos = DecoderSOS(model.decoder)
    decoder_sos.eval()
    programs["decoder_sos"] = export(
        decoder_sos,
        (),
        strict=False,
    )
    print("Exported decoder_sos")

    # Joint - static shapes since we process one timestep at a time
    f_proj = torch.randn(1, 1, 640)  # Already projected encoder output (single timestep)
    g_proj = torch.randn(1, 1, 640)  # Already projected decoder output
    programs["joint"] = export(
        JointAfterProjection(model.joint),
        (f_proj, g_proj),
        dynamic_shapes={"f": {}, "g": {}},
        strict=False,
    )
    print("Exported joint")

    # Joint projections
    # - Encoder projection: called once on full sequence [batch, time, hidden]
    # - Decoder projection: called per token [batch, 1, hidden]
    encoder_hidden = 1024
    pred_hidden = getattr(model.decoder, "pred_hidden", 640)

    programs["joint_project_encoder"] = export(
        JointProjectEncoder(model.joint),
        (torch.randn(1, 25, encoder_hidden),),
        dynamic_shapes={"f": {1: Dim("enc_time", min=1, max=60000)}},  # max ~60s audio after subsampling
        strict=False,
    )
    print("Exported joint_project_encoder")

    programs["joint_project_decoder"] = export(
        JointProjectDecoder(model.joint),
        (torch.randn(1, 1, pred_hidden),),
        dynamic_shapes={"g": {}},
        strict=False,
    )
    print("Exported joint_project_decoder")

    return programs


def lower_to_executorch(programs, output_dir, use_xnnpack=False):
    """Lower all ExportedPrograms to a single .pte file."""
    if use_xnnpack:
        print("\nLowering to ExecuTorch with XNNPack...")
        partitioner = [XnnpackPartitioner()]
    else:
        print("\nLowering to ExecuTorch (portable, no delegation)...")
        partitioner = []

    et_prog = to_edge_transform_and_lower(
        programs,
        partitioner=partitioner,
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
            _skip_dim_order=True,
        ),
    )
    et = et_prog.to_executorch(
        config=ExecutorchBackendConfig(
            extract_delegate_segments=True,
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        ),
    )

    # pte_path = os.path.join(output_dir, "parakeet_tdt.pte")
    # with open(pte_path, "wb") as f:
    #     f.write(et.buffer)
    # print(f"Saved {pte_path}")
    return et


def verify(model, programs, et_buffer=None):
    """Verify exported programs match eager model outputs within 1e-5."""
    print("\nVerifying exported programs...")

    # Encoder
    feat_in = getattr(model.encoder, "_feat_in", 80)
    audio = torch.randn(1, feat_in, 100)
    length = torch.tensor([100], dtype=torch.int64)
    eager_enc = model.encoder(audio_signal=audio, length=length)
    export_enc = programs["encoder"].module()(audio_signal=audio, length=length)
    assert torch.allclose(eager_enc[0], export_enc[0], atol=1e-5), "Encoder mismatch"
    assert torch.allclose(eager_enc[1], export_enc[1], atol=1e-5), "Encoder length mismatch"
    print("  encoder: OK")

    # Decoder over 2 timesteps - batch mode with states=None
    for t in range(2):
        targets = torch.randint(0, 100, (1, 1), dtype=torch.long)
        target_length = torch.tensor([1], dtype=torch.int64)

        eager_dec = model.decoder(targets=targets, target_length=target_length, states=None)
        export_dec = programs["decoder"].module()(targets=targets, target_length=target_length, states=None)

        assert torch.allclose(eager_dec[0], export_dec[0], atol=1e-5), f"Decoder output mismatch at t={t}"
    print("  decoder (2 timesteps): OK")

    # Joint
    enc_proj = torch.randn(1, 1, 640)
    dec_proj = torch.randn(1, 1, 640)
    eager_joint = model.joint.joint_after_projection(enc_proj, dec_proj)
    export_joint = programs["joint"].module()(enc_proj, dec_proj)
    assert torch.allclose(eager_joint, export_joint, atol=1e-5), "Joint mismatch"
    print("  joint: OK")

    # Joint projections
    enc_hidden = torch.randn(1, 25, 1024)
    eager_proj_enc = model.joint.project_encoder(enc_hidden)
    export_proj_enc = programs["joint_project_encoder"].module()(enc_hidden)
    assert torch.allclose(eager_proj_enc, export_proj_enc, atol=1e-5), "Joint project_encoder mismatch"
    print("  joint_project_encoder: OK")

    pred_hidden = getattr(model.decoder, "pred_hidden", 640)
    dec_hidden = torch.randn(1, 1, pred_hidden)
    eager_proj_dec = model.joint.project_prednet(dec_hidden)
    export_proj_dec = programs["joint_project_decoder"].module()(dec_hidden)
    assert torch.allclose(eager_proj_dec, export_proj_dec, atol=1e-5), "Joint project_decoder mismatch"
    print("  joint_project_decoder: OK")

    print("Exported programs verified!")
    verify_executorch(model, et_buffer)


def verify_executorch(model, et_buffer):
    """Verify ExecuTorch runtime outputs match eager model within 1e-5."""
    from executorch.runtime import Runtime

    print("\nVerifying ExecuTorch runtime...")
    runtime = Runtime.get()
    program = runtime.load_program(et_buffer)

    # Ensure model is in eval mode and use no_grad for deterministic behavior
    model.eval()

    with torch.no_grad():
        # Encoder - use consistent input
        torch.manual_seed(42)
        feat_in = getattr(model.encoder, "_feat_in", 80)
        audio = torch.randn(1, feat_in, 100)
        length = torch.tensor([100], dtype=torch.int64)

        # Compare inputs
        print(f"    input audio sum: {audio.sum().item():.6f}, length: {length.item()}")

        eager_enc = model.encoder(audio_signal=audio, length=length)
        method = program.load_method("encoder")
        et_enc = method.execute([audio, length])

        # Debug: show first few values
        print(f"    eager first 5: {eager_enc[0].flatten()[:5].tolist()}")
        print(f"    et    first 5: {et_enc[0].flatten()[:5].tolist()}")

    diff = (eager_enc[0] - et_enc[0]).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"    encoder max_diff: {max_diff:.2e}, mean_diff: {mean_diff:.2e}")
    assert torch.allclose(eager_enc[0], et_enc[0], atol=1e-5), f"ET Encoder mismatch (max_diff={max_diff:.2e})"
    print("  encoder: OK")

    # Decoder over 2 timesteps - exported with states=None
    method = program.load_method("decoder")
    for t in range(2):
        targets = torch.randint(0, 100, (1, 1), dtype=torch.long)
        target_length = torch.tensor([1], dtype=torch.int64)

        eager_dec = model.decoder(targets=targets, target_length=target_length, states=None)
        et_dec = method.execute([targets, target_length, None])

        assert torch.allclose(eager_dec[0], et_dec[0], atol=1e-5), f"ET Decoder output mismatch at t={t}"
    print("  decoder (2 timesteps): OK")

    # Joint
    enc_proj = torch.randn(1, 1, 640)
    dec_proj = torch.randn(1, 1, 640)
    eager_joint = model.joint.joint_after_projection(enc_proj, dec_proj)
    method = program.load_method("joint")
    et_joint = method.execute([enc_proj, dec_proj])
    assert torch.allclose(eager_joint, et_joint[0], atol=1e-5), "ET Joint mismatch"
    print("  joint: OK")

    # Joint projections
    enc_hidden = torch.randn(1, 25, 1024)
    eager_proj_enc = model.joint.project_encoder(enc_hidden)
    method = program.load_method("joint_project_encoder")
    et_proj_enc = method.execute([enc_hidden])
    assert torch.allclose(eager_proj_enc, et_proj_enc[0], atol=1e-5), "ET project_encoder mismatch"
    print("  joint_project_encoder: OK")

    pred_hidden = getattr(model.decoder, "pred_hidden", 640)
    dec_hidden = torch.randn(1, 1, pred_hidden)
    eager_proj_dec = model.joint.project_prednet(dec_hidden)
    method = program.load_method("joint_project_decoder")
    et_proj_dec = method.execute([dec_hidden])
    assert torch.allclose(eager_proj_dec, et_proj_dec[0], atol=1e-5), "ET project_decoder mismatch"
    print("  joint_project_decoder: OK")

    print("ExecuTorch runtime verified!")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./parakeet_tdt_exports")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--audio", type=str, help="Path to audio file for transcription test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    model = load_model()

    print("\nExporting components...")
    programs = export_all(model)

    # # Save individual .pt2 files
    # for name, ep in programs.items():
    #     torch.export.save(ep, os.path.join(args.output_dir, f"{name}.pt2"))
    # print(f"\nSaved {len(programs)} .pt2 files")

    # Lower all to single .pte
    et = lower_to_executorch(programs, args.output_dir)

    if not args.skip_verify:
        verify(model, programs, et.buffer)

    # Test transcription if audio file provided
    if args.audio:
        print("\n" + "=" * 60)
        print("Testing transcription...")
        print("=" * 60)

        print("\n[Eager PyTorch]")
        eager_text = transcribe_eager(args.audio, model)
        print(f"  Result: {eager_text}")

        print("\n[ExecuTorch Runtime]")
        et_text = transcribe_executorch(args.audio, model, et.buffer)
        print(f"  Result: {et_text}")

        if eager_text == et_text:
            print("\n✓ Transcriptions match!")
        else:
            print("\n✗ Transcriptions differ!")
            print(f"  Eager: {eager_text}")
            print(f"  ET:    {et_text}")

    print("\nDone!")


if __name__ == "__main__":
    main()
