from pathlib import Path


def main():
    path = Path("docs/mostlyright_assessment.md")
    if not path.exists():
        raise SystemExit("docs/mostlyright_assessment.md not found")
    print(path.read_text(encoding="utf-8")[:2000])


if __name__ == "__main__":
    main()
