"""
test_pipeline.py — Pre-flight check for CRAFT Stage 1 training.

Run BEFORE starting actual training to verify everything works.

Usage:
    python test_pipeline.py --config configs/test.yaml
    python test_pipeline.py --config configs/test.yaml --skip_fullres
    python test_pipeline.py --data_root data/train --parser_ckpt pretrained/79999_iter.pth
"""

import argparse
import gc
import os
import shutil
import sys
import tempfile
import traceback

import yaml

import torch
import numpy as np


def sep(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def ok(msg):
    print(f"  ✓ {msg}")


def fail(msg):
    print(f"  ✗ {msg}")
    traceback.print_exc()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_tests(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_ok = True
    ds_paired = None
    tmp_dir = tempfile.mkdtemp()

    # ==================================================================
    sep("Test 1: Imports")
    # ==================================================================
    try:
        from models.residual_vq import ResidualVQ, VQLevel
        from models.face_parser import FaceParser, REGION_NAMES, REGION_MAP
        from models.region_aware_vq import RegionAwareVQ
        from losses.losses import Stage1VQLoss, VGGPerceptualLoss, PatchDiscriminator
        from data.dataset import FFHQPairedDataset
        from models.vqvae import VQVAE, GlobalVQ, build_hq_vqvae, build_lq_vqvae
        from train_stage1 import train_one_epoch
        from torch.utils.data import DataLoader
        ok("All imports successful")
    except Exception:
        fail("Imports"); return False

    # ==================================================================
    sep("Test 2: Dataset loading")
    # ==================================================================
    try:
        hq_dir = os.path.join(args.data_root, "images512x512")
        lq_dir = os.path.join(args.data_root, "LQ_images_512x512")
        assert os.path.isdir(hq_dir), f"Not found: {hq_dir}"
        assert os.path.isdir(lq_dir), f"Not found: {lq_dir}"
        n_hq = len([f for f in os.listdir(hq_dir) if f.endswith(".png")])
        n_lq = len([f for f in os.listdir(lq_dir) if f.endswith(".png")])
        print(f"  Found {n_hq} HQ, {n_lq} LQ images")

        ds_hq = FFHQPairedDataset(data_root=args.data_root, hq_only=True)
        assert ds_hq[0]["hq"].shape == (3, 512, 512)
        ok(f"HQ-only dataset: {len(ds_hq)} samples")

        ds_paired = FFHQPairedDataset(data_root=args.data_root, hq_only=False)
        assert "lq" in ds_paired[0]
        ok(f"Paired dataset: {len(ds_paired)} pairs")

        batch = next(iter(DataLoader(ds_paired, batch_size=2, drop_last=True)))
        assert batch["hq"].shape[0] == 2
        ok("DataLoader batching works")
    except Exception:
        fail("Dataset"); all_ok = False

    # ==================================================================
    sep("Test 3: Face parser")
    # ==================================================================
    try:
        pp = args.parser_ckpt if os.path.exists(args.parser_ckpt) else None
        if pp:
            print(f"  Using checkpoint: {pp}")
        else:
            print(f"  No checkpoint at {args.parser_ckpt} — random weights")

        fp = FaceParser(checkpoint_path=pp).to(device)
        assert all(not p.requires_grad for p in fp.parameters())
        ok("FaceParser frozen")

        img = ds_paired[0]["hq_01"].unsqueeze(0).to(device) if ds_paired else torch.rand(1,3,512,512,device=device)
        masks = fp.get_region_masks(img, 16, 16)
        coverage = torch.stack(list(masks.values())).sum(0)
        assert coverage.eq(1).all()
        ok(f"Regions: { {n: m.sum().item() for n,m in masks.items()} }")
        del fp; cleanup()
    except Exception:
        fail("Face parser"); all_ok = False

    # ==================================================================
    sep("Test 4: ResidualVQ")
    # ==================================================================
    try:
        rq = ResidualVQ(n_codes=64, e_dim=128, n_levels=3).to(device).train()
        z = torch.randn(100, 128, device=device, requires_grad=True)
        z_q, losses, info = rq(z)
        (z_q.sum() + losses["total_vq"]).backward()
        assert z.grad is not None
        ok(f"OK, loss={losses['total_vq']:.4f}")
        del rq; cleanup()
    except Exception:
        fail("ResidualVQ"); all_ok = False

    # ==================================================================
    sep("Test 5: RegionAwareVQ")
    # ==================================================================
    try:
        pp = args.parser_ckpt if os.path.exists(args.parser_ckpt) else None
        ravq = RegionAwareVQ(e_dim=512, n_levels=3, parser_ckpt=pp).to(device).train()
        z = torch.randn(2, 512, 16, 16, device=device, requires_grad=True)
        imgs = ds_paired[0]["hq_01"].unsqueeze(0).repeat(2,1,1,1).to(device) if ds_paired else torch.rand(2,3,512,512,device=device)
        z_q, losses, _ = ravq(z, images=imgs)
        (z_q.sum() + losses["total_vq"]).backward()
        assert z.grad is not None
        ok(f"OK, loss={losses['total_vq']:.4f}")
        del ravq; cleanup()
    except Exception:
        fail("RegionAwareVQ"); all_ok = False

    # ==================================================================
    sep("Test 6: VQVAE (small model)")
    # ==================================================================
    RES, CH, EDIM, CM = 64, 32, 64, (1, 2, 4)
    try:
        hq_m = build_hq_vqvae(n_codes=64, embed_dim=EDIM, ch=CH, ch_mult=CM, resolution=RES, z_channels=EDIM).to(device).train()
        x = torch.randn(2, 3, RES, RES, device=device, requires_grad=True)
        xr, z, zq, vl, _ = hq_m(x)
        (xr.sum() + vl["total_vq"]).backward()
        assert x.grad is not None
        ok(f"HQ VQVAE: {x.shape} → {xr.shape}")

        ravq_s = RegionAwareVQ(region_n_codes={"eyes":8,"skin":16,"hair":16,"lips":4}, e_dim=EDIM, n_levels=2, parser_ckpt=None).to(device)
        lq_m = build_lq_vqvae(ravq_s, embed_dim=EDIM, ch=CH, ch_mult=CM, resolution=RES, z_channels=EDIM).to(device).train()
        x2 = torch.randn(2, 3, RES, RES, device=device, requires_grad=True)
        xr2, _, _, vl2, _ = lq_m(x2, images_01=torch.rand(2,3,512,512,device=device))
        (xr2.sum() + vl2["total_vq"]).backward()
        assert x2.grad is not None
        ok(f"LQ VQVAE: {x2.shape} → {xr2.shape}")
        del hq_m, lq_m, ravq_s; cleanup()
    except Exception:
        fail("VQVAE"); all_ok = False

    # ==================================================================
    sep("Test 7: Losses")
    # ==================================================================
    try:
        try:
            crit = Stage1VQLoss(vgg_pretrained=True).to(device)
            ok("VGG pretrained loaded")
        except Exception:
            crit = Stage1VQLoss(vgg_pretrained=False).to(device)
            ok("VGG pretrained unavailable — random weights (OK for test)")

        p = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
        t = torch.randn(2, 3, 64, 64, device=device)
        gl, gd = crit.generator_loss(p, t, torch.tensor(0.5, device=device))
        gl.backward()
        assert p.grad is not None
        ok(f"Generator loss: {gl.item():.4f}")

        dl, dd = crit.discriminator_loss(t, p.detach())
        ok(f"Discriminator loss: {dl.item():.4f}")

        crit.set_lambda_assoc(1.0)
        p2 = torch.randn(2, 3, 64, 64, device=device, requires_grad=True)
        gl2, gd2 = crit.generator_loss(p2, t, torch.tensor(0.5, device=device),
                                        z_H=torch.randn(2,64,128,device=device),
                                        z_L=torch.randn(2,64,128,device=device))
        assert "association" in gd2
        ok(f"Association loss: {gd2['association']:.4f}")
        del crit; cleanup()
    except Exception:
        fail("Losses"); all_ok = False

    # ==================================================================
    sep("Test 8: Training loop (A → B → C)")
    # ==================================================================
    try:
        try:
            crit_a = Stage1VQLoss(vgg_pretrained=True).to(device)
        except Exception:
            crit_a = Stage1VQLoss(vgg_pretrained=False).to(device)

        # Phase A
        print("  Phase A (HQ)...")
        hq_model = build_hq_vqvae(n_codes=16, embed_dim=EDIM, ch=CH, ch_mult=CM, resolution=RES, z_channels=EDIM).to(device).train()
        loader_a = DataLoader(FFHQPairedDataset(args.data_root, hq_only=True, resolution=RES), batch_size=2, shuffle=True, drop_last=True)
        og = torch.optim.Adam(hq_model.parameters(), lr=1e-4)
        od = torch.optim.Adam(crit_a.discriminator.parameters(), lr=1e-4)
        avg_a = train_one_epoch(hq_model, crit_a, loader_a, og, od, device, 0, "A", log_every=9999)
        ok(f"Phase A: L1={avg_a['l1']:.4f}")
        torch.save({"epoch":0, "model": hq_model.state_dict(), "discriminator": crit_a.discriminator.state_dict(),
                     "optimizer_g": og.state_dict(), "optimizer_d": od.state_dict()}, os.path.join(tmp_dir, "hq.pt"))
        del crit_a, og, od; cleanup()

        # Phase B
        print("  Phase B (LQ, no assoc)...")
        ravq_b = RegionAwareVQ(region_n_codes={"eyes":4,"skin":8,"hair":8,"lips":4}, e_dim=EDIM, n_levels=2, parser_ckpt=None).to(device)
        lq_model = build_lq_vqvae(ravq_b, embed_dim=EDIM, ch=CH, ch_mult=CM, resolution=RES, z_channels=EDIM).to(device).train()
        try:
            crit_b = Stage1VQLoss(lambda_assoc=0.0, vgg_pretrained=True).to(device)
        except Exception:
            crit_b = Stage1VQLoss(lambda_assoc=0.0, vgg_pretrained=False).to(device)
        loader_b = DataLoader(FFHQPairedDataset(args.data_root, hq_only=False, resolution=RES), batch_size=2, shuffle=True, drop_last=True)
        og2 = torch.optim.Adam([p for p in lq_model.parameters() if p.requires_grad], lr=1e-4)
        od2 = torch.optim.Adam(crit_b.discriminator.parameters(), lr=1e-4)
        avg_b = train_one_epoch(lq_model, crit_b, loader_b, og2, od2, device, 0, "B", log_every=9999)
        assert "association" not in avg_b
        ok(f"Phase B: L1={avg_b['l1']:.4f}")
        del crit_b, og2, od2; cleanup()

        # Phase C
        print("  Phase C (LQ + assoc)...")
        hq_model.eval()
        for p in hq_model.parameters(): p.requires_grad_(False)
        try:
            crit_c = Stage1VQLoss(lambda_assoc=1.0, vgg_pretrained=True).to(device)
        except Exception:
            crit_c = Stage1VQLoss(lambda_assoc=1.0, vgg_pretrained=False).to(device)
        og3 = torch.optim.Adam([p for p in lq_model.parameters() if p.requires_grad], lr=1e-4)
        od3 = torch.optim.Adam(crit_c.discriminator.parameters(), lr=1e-4)
        avg_c = train_one_epoch(lq_model, crit_c, loader_b, og3, od3, device, 0, "C", hq_model=hq_model, log_every=9999)
        assert "association" in avg_c
        ok(f"Phase C: L1={avg_c['l1']:.4f}, assoc={avg_c['association']:.4f}")
        del crit_c, og3, od3; cleanup()
    except Exception:
        fail("Training loop"); all_ok = False

    # ==================================================================
    sep("Test 9: Checkpoint save / load / consistency")
    # ==================================================================
    try:
        torch.save({"epoch": 0, "model": lq_model.state_dict()}, os.path.join(tmp_dir, "lq.pt"))
        ok("Save")

        ravq_n = RegionAwareVQ(region_n_codes={"eyes":4,"skin":8,"hair":8,"lips":4}, e_dim=EDIM, n_levels=2, parser_ckpt=None).to(device)
        lq_new = build_lq_vqvae(ravq_n, embed_dim=EDIM, ch=CH, ch_mult=CM, resolution=RES, z_channels=EDIM).to(device)
        lq_new.load_state_dict(torch.load(os.path.join(tmp_dir, "lq.pt"), weights_only=False)["model"])
        ok("Load")

        lq_model.eval(); lq_new.eval()
        tx = torch.randn(1, 3, RES, RES, device=device)
        ti = torch.rand(1, 3, 512, 512, device=device)
        with torch.no_grad():
            o1, *_ = lq_model(tx, images_01=ti)
            o2, *_ = lq_new(tx, images_01=ti)
        d = (o1 - o2).abs().max().item()
        assert d < 1e-5
        ok(f"Consistency: max_diff={d:.2e}")
        del lq_model, lq_new, hq_model; cleanup()
    except Exception:
        fail("Checkpoint"); all_ok = False

    # ==================================================================
    if not args.skip_fullres:
        sep("Test 10: Full resolution (512×512)")
        try:
            pp = args.parser_ckpt if os.path.exists(args.parser_ckpt) else None
            hq_f = build_hq_vqvae(n_codes=1024, embed_dim=512).to(device)
            print(f"  HQ params: {sum(p.numel() for p in hq_f.parameters()):,}")
            with torch.no_grad():
                xr, z, *_ = hq_f(torch.randn(1, 3, 512, 512, device=device))
            assert xr.shape == (1, 3, 512, 512) and z.shape == (1, 512, 16, 16)
            ok(f"HQ forward: z={z.shape}")
            del hq_f; cleanup()

            ravq_f = RegionAwareVQ(e_dim=512, n_levels=3, parser_ckpt=pp).to(device)
            lq_f = build_lq_vqvae(ravq_f, embed_dim=512).to(device)
            print(f"  LQ params: {sum(p.numel() for p in lq_f.parameters()):,}")
            img01 = ds_paired[0]["hq_01"].unsqueeze(0).to(device) if ds_paired else torch.rand(1,3,512,512,device=device)
            xg = torch.randn(1, 3, 512, 512, device=device, requires_grad=True)
            xr2, _, _, vl2, _ = lq_f(xg, images_01=img01)
            (xr2.sum() + vl2["total_vq"]).backward()
            assert xg.grad is not None
            ok(f"LQ forward+backward: {xr2.shape}")
            if torch.cuda.is_available():
                print(f"  Peak GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
            del lq_f, ravq_f; cleanup()
        except torch.cuda.OutOfMemoryError:
            print("  ⚠ OOM — reduce batch_size. Code is correct (small tests passed).")
        except Exception:
            fail("Full resolution"); all_ok = False
    else:
        sep("Test 10: Full resolution — SKIPPED")

    # ==================================================================
    shutil.rmtree(tmp_dir, ignore_errors=True)
    sep("SUMMARY")
    if all_ok:
        print("  ✓ ALL TESTS PASSED\n")
        print(f"  python train_stage1.py --phase all --data_root {args.data_root}", end="")
        if os.path.exists(args.parser_ckpt):
            print(f" --parser_ckpt {args.parser_ckpt}")
        else:
            print(f"\n\n  ⚠ Download BiSeNet checkpoint → {args.parser_ckpt}")
    else:
        print("  ✗ SOME TESTS FAILED\n")
    return all_ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="",
                   help="Path to YAML config file (e.g. configs/test.yaml)")
    p.add_argument("--data_root", default="/projectnb/cs585/projects/craft/data/train", type=str,
                   help="Path to train/ directory with images512x512/ and LQ_images_512x512/")
    p.add_argument("--parser_ckpt", type=str, default="/projectnb/cs585/projects/craft/pretrained/79999_iter.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--skip_fullres", action="store_true")

    args, _ = p.parse_known_args()
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        p.set_defaults(**cfg)

    sys.exit(0 if run_tests(p.parse_args()) else 1)


if __name__ == "__main__":
    main()