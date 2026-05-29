import pandas as pd
import typer

from weather_tmax_bot.models.train import train_quantile_model


def main(
    airport: str = typer.Option("EDDM"),
    dataset_path: str = typer.Option("data/processed/training_dataset.parquet"),
    model_version: str = typer.Option("quantile_mvp"),
):
    dataset = pd.read_parquet(dataset_path)
    if "leakage_check_passed" in dataset.columns:
        dataset = dataset[dataset["leakage_check_passed"] == True].copy()
    if "airport_icao" in dataset.columns:
        dataset = dataset[dataset["airport_icao"] == airport].copy()
    train_quantile_model(dataset, model_version=model_version)
    print(f"Saved quantile model to data/models/{model_version}.joblib")


if __name__ == "__main__":
    typer.run(main)
