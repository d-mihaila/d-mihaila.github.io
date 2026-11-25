import hashlib
import json
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
ROWS = [
    (
        "gray-et-al-1989",
        3,
        "gray-fig3-d-iii",
        (0.510, 0.797, 0.706, 0.945),
        "Figure 3d, condition III",
        "Published version",
        "© 1989 Nature Publishing Group; all rights reserved",
        "https://doi.org/10.1038/338334a0",
    ),
    (
        "fries-et-al-2001",
        3,
        "fries-fig1-j",
        (0.495, 0.580, 0.642, 0.688),
        "Figure 1J",
        "Published version",
        "© 2001 American Association for the Advancement of Science; all rights reserved",
        "https://doi.org/10.1126/science.1055465",
    ),
    (
        "sohal-et-al-2009",
        3,
        "sohal-fig2-e",
        (0.055, 0.380, 0.281, 0.506),
        "Figure 2e",
        "Published version",
        "© 2009 Macmillan Publishers Limited; all rights reserved",
        "https://doi.org/10.1038/nature07991",
    ),
    (
        "drebitz-et-al-2025",
        3,
        "drebitz-fig2-b-curve",
        (0.363, 0.223, 0.586, 0.433),
        "Figure 2B phase curve, excluding inset",
        "Published version",
        "CC BY 4.0",
        "https://doi.org/10.1038/s41467-025-62732-8",
    ),
]
metadata = []
for name, page, filename, box, panel, version, rights, doi in ROWS:
    source = ROOT / "papers" / (name + ".pdf")
    document = pymupdf.open(source)
    original = document[page - 1]
    width, height = original.rect.width, original.rect.height
    clip = pymupdf.Rect(
        box[0] * width, box[1] * height, box[2] * width, box[3] * height
    )
    # Tighten vertical whitespace for the three classic plots, preserving all marks.
    classic = filename in {"gray-fig3-d-iii", "fries-fig1-j", "sohal-fig2-e"}
    if classic:
        probe = original.get_pixmap(matrix=pymupdf.Matrix(4, 4), clip=clip, alpha=False)
        samples = probe.samples
        occupied = [
            y
            for y in range(probe.height)
            if min(samples[y * probe.stride : y * probe.stride + probe.width * 3]) < 250
        ]
        if not occupied:
            raise ValueError(f"No content found in {filename}")
        clip.y0 += max(0, occupied[0] - 1) / 4
        clip.y1 -= max(0, probe.height - occupied[-1] - 2) / 4
    horizontal_margin = 18
    vertical_margin = 4 if classic else 18
    padded = pymupdf.open()
    canvas = padded.new_page(
        width=clip.width + 2 * horizontal_margin,
        height=clip.height + 2 * vertical_margin,
    )
    canvas.show_pdf_page(
        pymupdf.Rect(
            horizontal_margin,
            vertical_margin,
            horizontal_margin + clip.width,
            vertical_margin + clip.height,
        ),
        document,
        page - 1,
        clip=clip,
    )
    canvas.get_pixmap(matrix=pymupdf.Matrix(4, 4), alpha=False).save(
        OUT / (filename + ".png")
    )
    metadata.append(
        dict(
            file=filename + ".png",
            source=str(source.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            doi=doi,
            pdf_page=page,
            panel=panel,
            version=version,
            rights=rights,
            changes="Single source plot cropped; vertical whitespace trimmed where applicable; white margins retained; rasterised at 4× resolution. No data redrawn.",
            normalised_crop=box,
            rendered_crop=[
                clip.x0 / width,
                clip.y0 / height,
                clip.x1 / width,
                clip.y1 / height,
            ],
            white_padding_pixels={
                "horizontal": horizontal_margin * 4,
                "vertical": vertical_margin * 4,
            },
        )
    )
(OUT / "sources.json").write_text(
    json.dumps({"extracted_on": "2026-11-24", "panels": metadata}, indent=2) + "\n"
)
print(f"Extracted {len(ROWS)} standalone plots with white margins.")
