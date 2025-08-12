import time
import torch

import env


def print_stats(name: str, t: torch.Tensor) -> None:
    t_cpu = t.detach().to(torch.float32).cpu()
    print(f"{name}: shape={tuple(t.shape)}, device={t.device}, dtype={t.dtype}, "
          f"mean={t_cpu.mean().item():.4f}, std={t_cpu.std().item():.4f}, "
          f"min={t_cpu.min().item():.4f}, max={t_cpu.max().item():.4f}")


def main() -> None:
    torch.manual_seed(42)

    # Small batch for readable prints
    batch_size = 4
    print("Initializing CUDA env...")
    codebook = env.initialise_global_codebook(batch_size)
    print_stats("Initial codebook", codebook)

    rows = int(codebook.shape[1])
    cols = int(codebook.shape[2])

    # Environment layout
    amp_L = 16
    amp_J = 6
    amp_N = amp_L * (2 ** amp_J)
    assert amp_N == cols, f"N mismatch: {amp_N} vs cols {cols}"

    # Use identity sensing to make y interpretation clear (n = rows)
    amp_n = rows
    S_identity = torch.eye(amp_n, rows, dtype=torch.float64, device="cpu").contiguous()
    print_stats("S (identity)", S_identity)

    # Dummy actions (one per env), on CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actions = torch.randn(batch_size, rows, device=device)
    print_stats("Actions (pre-step)", actions)

    # Snapshot last column of env 0 before step
    with torch.no_grad():
        cb0_before = env.get_codebook()[0].detach().cpu().clone()
        last_col_before = cb0_before[:, -1].clone()
    print_stats("Env0 last_col (before)", last_col_before)

    # Run one AMP step
    T_max = 8
    tol = 1e-6
    P_hat = 1.0
    print("\nRunning env.step_amp() ...")
    t0 = time.time()
    hit_rates, rewards, dones = env.step_amp(
        actions=actions,
        sensing_matrix=S_identity,
        n=amp_n,
        N=amp_N,
        T_max=T_max,
        tol=tol,
        P_hat=P_hat,
    )
    dt = time.time() - t0
    print(f"step_amp finished in {dt:.3f}s")
    print_stats("Hit rates", hit_rates)
    print_stats("Rewards", rewards)
    print_stats("Dones", dones)

    # Snapshot last column of env 0 after step
    with torch.no_grad():
        cb0_after = env.get_codebook()[0].detach().cpu().clone()
        last_col_after = cb0_after[:, -1].clone()
    print_stats("Env0 last_col (after)", last_col_after)

    # Show a small diff between last columns to confirm codebook update
    diff = (last_col_after - last_col_before)
    print_stats("Env0 last_col diff", diff)
    print("\nSample entries from last_col (before → after):")
    for idx in [0, 1, 2, 100, 200, 300, 400, rows - 1]:
        print(
            f"  idx {idx:3d}: {last_col_before[idx].item(): .4f} → {last_col_after[idx].item(): .4f}"
        )

    # Optional: run a couple more steps to see stats evolve
    print("\nRunning 2 more AMP steps...")
    for s in range(2):
        actions = torch.randn(batch_size, rows, device=device)
        hit_rates, rewards, dones = env.step_amp(
            actions=actions,
            sensing_matrix=S_identity,
            n=amp_n,
            N=amp_N,
            T_max=T_max,
            tol=tol,
            P_hat=P_hat,
        )
        print(
            f"Step {s+1}: avg_hit={hit_rates.float().mean().item():.4f}, "
            f"max_hit={hit_rates.float().max().item():.4f}"
        )


if __name__ == "__main__":
    main()


