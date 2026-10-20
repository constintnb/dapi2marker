import numpy as np
from PIL import Image
import torch

from dapi2marker.data.dataset import PairedDataset, make_sampler
from dapi2marker.data.geometry import AdjacentCrop


def _tiles(tmp_path):
    records = []
    for tissue, value in (("colon", 20), ("liver", 80)):
        for row in range(2):
            for col in range(2):
                filename = f"{tissue}_ROI001_{row}_{col}.png"
                label = "label_" + filename
                image = np.full((256, 256), value + row * 2 + col, dtype=np.uint8)
                Image.fromarray(image).save(tmp_path / filename)
                Image.fromarray(image + 10).save(tmp_path / label)
                records.append(
                    {
                        "source_root": str(tmp_path),
                        "dapi_path": filename,
                        "target_path": label,
                        "image_size": [256, 256],
                        "sample_id": f"{tissue}:ROI001:{row}:{col}:CD45RO",
                        "group_id": f"{tissue}:ROI001",
                        "tissue": tissue,
                        "roi": "ROI001",
                        "row": row,
                        "col": col,
                    }
                )
    return records


def test_crop_namespace_pairing_and_validation_neighbors_excluded(tmp_path):
    records = _tiles(tmp_path)
    options = {
        "enabled": True,
        "layout": "roi_row_col_stride256",
        "tissues": ["colon", "liver"],
        "probability": 1,
        "max_shift_pixels": 8,
    }
    crop = AdjacentCrop(records, options)
    for record in records:
        (image, label), _ = crop.sample(record, 0)
        assert np.array_equal(label, image + 10)
        assert image.min() >= (80 if record["tissue"] == "liver" else 20)
        assert image.max() <= (83 if record["tissue"] == "liver" else 23)
    anchor = records[4]
    assert len(crop.plan(anchor, 8, 8)) == 4
    restricted = AdjacentCrop([anchor], options)
    assert restricted.plan(anchor, 8, 8) is None
    (image, label), meta = restricted.sample(anchor, 0)
    assert not meta["shifted"] and meta["source_tiles"] == 1
    assert np.array_equal(label, image + 10)


def test_paired_augmentation_repeatable_and_sampler_balanced(tmp_path):
    records = _tiles(tmp_path)
    # 4 colon tiles and just one liver tile should still have equal expected mass.
    dataset = PairedDataset(records[:5], augment=True)
    dataset.set_epoch(2)
    one, two = dataset[0], dataset[0]
    torch.testing.assert_close(one["dapi"], two["dapi"], rtol=0, atol=0)
    torch.testing.assert_close(
        one["targets"] - one["dapi"], torch.full_like(one["dapi"], 10 / 127.5)
    )
    sampler = make_sampler(dataset, "tissue_balanced", torch.Generator().manual_seed(1))
    assert sampler.weights[:4].sum() == sampler.weights[4:].sum()
