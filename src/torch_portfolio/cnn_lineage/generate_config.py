from pathlib import Path

import typer

conf_dir = Path("src/torch_portfolio/cnn_lineage/conf/model")


def main(model: str):
    conf_dir.mkdir(parents=True, exist_ok=True)
    file_path = conf_dir / f"{model}.yaml"
    if not file_path.exists():
        _ = file_path.write_text(f"name: {model}")
        print(f"Created config file for model: {model}")
        typer.echo(f"Successfully created configuration file at: {file_path}")
    else:
        typer.echo(f"File already exists at {file_path}")


if __name__ == "__main__":
    typer.run(main)
