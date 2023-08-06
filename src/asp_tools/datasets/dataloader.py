import torch


def test_collate_fn(batch):
    batched_mixture, batched_sources = None, None
    batched_segment_ID = []

    for mixture, sources, segmend_ID in batch:
        mixture = mixture.unsqueeze(dim=0)
        sources = sources.unsqueeze(dim=0)

        if batched_mixture is None:
            batched_mixture = mixture
            batched_sources = sources
        else:
            batched_mixture = torch.cat([batched_mixture, mixture], dim=0)
            batched_sources = torch.cat([batched_sources, sources], dim=0)

        batched_segment_ID.append(segmend_ID)

    return batched_mixture, batched_sources, batched_segment_ID
