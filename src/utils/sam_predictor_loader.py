def load_sam_predictor(checkpoint_path: str, device: str):
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry['vit_h'](checkpoint=checkpoint_path)
    sam.to(device)
    print("SAM-H loaded.")
    return SamPredictor(sam)