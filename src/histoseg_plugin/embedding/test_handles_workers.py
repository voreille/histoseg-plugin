from torch.utils.data import DataLoader

ds = WholeSlidePatchH5("patches/SLIDE.h5", "slides/SLIDE.svs", tx)
loader = DataLoader(ds, batch_size=32, num_workers=2, persistent_workers=True)

for i, batch in enumerate(loader):
    if i < 3:
        # inspect per-worker PIDs and object ids
        print("pid", os.getpid(), "h5 id", id(ds.h5), "wsi id", id(ds.wsi))
