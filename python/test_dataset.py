from dataset_div2k import DIV2KPatchDataset

# IMPORTANT: dataset is ONE LEVEL ABOVE python/
ds = DIV2KPatchDataset("../dataset", split="train")

lr, hr = ds.sample_batch(2)

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)
