from __future__ import annotations

import typer

from weather_tmax_bot.evaluation.first_analysis import write_first_analysis_report


def main(
    json_path: str = typer.Option("data/reports/first_analysis.json"),
    markdown_path: str = typer.Option("docs/first_analysis.md"),
):
    json_output, markdown_output = write_first_analysis_report(json_path=json_path, markdown_path=markdown_path)
    print(f"Wrote {json_output}")
    print(f"Wrote {markdown_output}")


if __name__ == "__main__":
    typer.run(main)
