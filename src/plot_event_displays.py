import h5py
import matplotlib.pyplot as plt


for _, row in sample.iterrows():

    event_id = int(row["event_id"])

    pred = row["prediction"]

    true = row["label"]

    event = events[event_id]

    proj = np.sum(event, axis=0)

    plt.figure(figsize=(6,6))

    plt.imshow(proj, origin="lower")

    plt.title(
        f"true={true} | pred={pred}"
    )

    plt.colorbar()

    plt.savefig(...)
