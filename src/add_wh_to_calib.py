"""
add_wh_to_calib.py — Adiciona 'w', 'h' (e opcionalmente 'roi') a arquivos
.npz de calibração que não têm essas chaves.

Exemplos:

Todos os .npz da pasta:
    python add_wh_to_calib.py calib/ --width 1920 --height 1080

Usando glob:
    python add_wh_to_calib.py calib/ --width 1920 --height 1080 --glob "calib_rt*.npz"

Arquivos específicos:
    python add_wh_to_calib.py calib/ \
        --width 1920 --height 1080 \
        --pattern "calib_rt{i}.npz" \
        --ids 0,1,2,3
"""

import argparse
import glob
import os
import shutil

import numpy as np


def add_wh(npz_path: str, width: int, height: int, roi=None, dry_run=False):
    if not os.path.exists(npz_path):
        print(f"  [erro] Arquivo não encontrado: {npz_path}")
        return

    data = dict(np.load(npz_path, allow_pickle=True))

    already_has = "w" in data and "h" in data
    if already_has:
        print(f"  [skip] {npz_path} já tem w/h ({int(data['w'])}x{int(data['h'])})")
        return

    data["w"] = np.array(width)
    data["h"] = np.array(height)

    if roi is not None:
        data["roi"] = np.array(roi)
    else:
        data["roi"] = np.array([0, 0, width, height])

    if dry_run:
        print(
            f"  [dry-run] {npz_path} -> adicionaria "
            f"w={width}, h={height}, roi={data['roi'].tolist()}"
        )
        return

    backup_path = npz_path + ".bak"
    if not os.path.exists(backup_path):
        shutil.copy2(npz_path, backup_path)

    np.savez(npz_path, **data)

    print(
        f"  [ok] {npz_path} -> "
        f"w={width}, h={height}, roi={data['roi'].tolist()} "
        f"(backup em {backup_path})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Adiciona w/h/roi a arquivos .npz de calibração."
    )

    parser.add_argument(
        "calib_dir",
        type=str,
        help="Pasta contendo os arquivos .npz."
    )

    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="Largura da imagem."
    )

    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="Altura da imagem."
    )

    parser.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI no formato x,y,w,h. Se omitido usa a imagem inteira."
    )

    parser.add_argument(
        "--glob",
        type=str,
        default="*.npz",
        help="Padrão glob para localizar arquivos (default: *.npz)."
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help='Padrão com {i}, ex.: "calib_rt{i}.npz".'
    )

    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Lista de IDs separados por vírgula. Ex.: 0,1,2,3"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra o que seria feito sem modificar os arquivos."
    )

    args = parser.parse_args()

    roi = None
    if args.roi:
        roi = [int(v) for v in args.roi.split(",")]
        if len(roi) != 4:
            raise ValueError("--roi deve ter formato x,y,w,h")

    # Seleção dos arquivos
    if args.pattern is not None or args.ids is not None:
        if args.pattern is None or args.ids is None:
            parser.error("--pattern e --ids devem ser usados juntos.")

        ids = [int(i.strip()) for i in args.ids.split(",") if i.strip()]

        npz_files = [
            os.path.join(args.calib_dir, args.pattern.format(i=i))
            for i in ids
        ]
    else:
        npz_files = sorted(
            glob.glob(os.path.join(args.calib_dir, args.glob))
        )

    if not npz_files:
        print("Nenhum arquivo encontrado.")
        return

    print(f"Encontrados {len(npz_files)} arquivos.")

    for npz_path in npz_files:
        add_wh(
            npz_path,
            args.width,
            args.height,
            roi=roi,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()