from pathlib import Path
import subprocess


def test_converter_expands_four_fitted_pairs_to_gpumd_matrix(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "fitted_IWP_parameters.dat"
    source.write_text(
        "\n".join(
            " ".join(str(100 * row + column) for column in range(1, 10))
            for row in range(1, 5)
        ) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "iwp_Mo6S6_Mo6Se6.txt"
    subprocess.run(
        [
            "python3",
            str(root / "tools" / "iwp" / "convert_fitted_iwp_to_gpumd.py"),
            str(source),
            str(output),
            "--wire-group-method", "2",
            "--rcut-global", "12.0",
            "--rcut-normal", "6.0",
        ],
        cwd=root,
        check=True,
    )
    raw_lines = [line.strip() for line in output.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert not any(line.startswith("#") for line in raw_lines), "fscanf-based GPUMD parser cannot skip comments"
    lines = raw_lines
    assert lines[0] == "nep_iwp 3 Mo S Se"
    assert lines[1] == "2"
    rows = [[float(value) for value in line.split()] for line in lines[2:]]
    assert len(rows) == 9
    assert all(len(row) == 12 for row in rows)
    # Row-major matrix: Mo-Mo, Mo-S, Mo-Se, S-Mo, S-S, S-Se, Se-Mo, Se-S, Se-Se.
    assert rows[0][:9] == [101.0 + i for i in range(9)]
    assert rows[1][:9] == [201.0 + i for i in range(9)]
    assert rows[2][:9] == [301.0 + i for i in range(9)]
    assert rows[3][:9] == rows[1][:9]
    assert rows[5][:9] == [401.0 + i for i in range(9)]
    assert rows[6][:9] == rows[2][:9]
    assert rows[7][:9] == rows[5][:9]
    assert all(row[9:] == [1.0, 6.0, 12.0] for row in rows)
