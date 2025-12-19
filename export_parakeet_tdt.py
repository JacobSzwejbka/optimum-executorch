#!/usr/bin/env python3
"""Export nvidia/parakeet-tdt-0.6b-v3 components to ExecuTorch."""

import os

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
from executorch.exir.passes import MemoryPlanningPass
from torch.export import Dim, export


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

    # Decoder - token-by-token, no dynamic shapes needed
    targets = torch.zeros(1, 1, dtype=torch.long)
    target_length = torch.tensor([1], dtype=torch.int64)
    programs["decoder"] = export(
        model.decoder,
        (),
        kwargs={"targets": targets, "target_length": target_length, "states": None},
        dynamic_shapes={"targets": {}, "target_length": {}, "states": None},
        strict=False,
    )
    print("Exported decoder")

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

    # Decoder over 2 timesteps - exported with states=None, so state is internal
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
        et_dec = method.execute([targets, target_length])

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

    print("\nDone!")


if __name__ == "__main__":
    main()
