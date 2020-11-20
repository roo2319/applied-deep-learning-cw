from coursework import Salicon, visualise
import torch
from torch.utils.data import DataLoader


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    test_dataset = Salicon(
            "train.pkl"
        )

    val_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=128,
        num_workers=1,
        pin_memory=True,
    )
    model = torch.load("model.pkl")
    results = {"preds": [], "gts": []}
    total_loss = 0
    model.eval()

    # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for batch, gts in val_loader:
            batch = batch.to(device)
            gts = gts.to(device)
            logits = model(batch)
            preds = logits.cpu().numpy()
            gts = gts.cpu().numpy()
            results["preds"].extend(list(preds))
            results["gts"].extend(list(gts))
            break
    visualise(results["preds"],test_dataset)

if __name__ == '__main__':
    main()
    